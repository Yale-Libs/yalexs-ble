from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import replace
from typing import Any, TypeVar, cast

from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakDBusError, BleakError
from bleak_retry_connector import (
    BLEAK_RETRY_EXCEPTIONS,
    MAX_CONNECT_ATTEMPTS,
    BleakNotFoundError,
    BLEDevice,
    get_device,
)
from lru import LRU  # pylint: disable=no-name-in-module

from .const import (
    APPLE_MFR_ID,
    HAP_ENCRYPTED_FIRST_BYTE,
    HAP_FIRST_BYTE,
    YALE_MFR_ID,
    AuthState,
    AutoLockMode,
    AutoLockState,
    BatteryState,
    ConnectionInfo,
    DoorStatus,
    LockInfo,
    LockState,
    LockStateValue,
    LockStatus,
)
from .lock import Lock
from .session import (
    AuthError,
    BluetoothError,
    DisconnectedError,
    NoAdvertisementError,
    ResponseError,
    YaleXSBLEError,
)
from .util import asyncio_timeout, is_disconnected_error, local_name_is_unique

_LOGGER = logging.getLogger(__name__)

# Advertisement debugger (this one is quite noisy so it has its only logger)
_ADV_LOGGER = logging.getLogger("yalexs_ble_adv")

WrapFuncType = TypeVar("WrapFuncType", bound=Callable[..., Any])

NEVER_TIME = -86400.0

DEFAULT_ATTEMPTS = 4

# How long to wait to disconnect after an operation
DISCONNECT_DELAY = 5.1

# How long to wait to disconnect after an operation if there is a pending update
DISCONNECT_DELAY_PENDING_UPDATE = 12.5

RESYNC_DELAY = 0.01

KEEP_ALIVE_TIME = 25.0  # Lock will disconnect after 30 seconds of inactivity

# Number of seconds to wait after the first connection
# to disconnect to free up the bluetooth adapter.
FIRST_CONNECTION_DISCONNECT_TIME = 2.1

# After a lock operation we need to wait for the lock to
# update its state or it will return a stale state.
LOCK_STALE_STATE_DEBOUNCE_DELAY = 6.1

# How long to wait before processing an advertisement change
ADV_UPDATE_COALESCE_SECONDS = 0.05

# How long to wait before processing the first update
FIRST_UPDATE_COALESCE_SECONDS = 0.01

# How long to wait before processing a HomeKit advertisement change
HK_UPDATE_COALESCE_SECONDS = 0.025

# How long to wait before processing a manual update request
MANUAL_UPDATE_COALESCE_SECONDS = 0.05

# How long to wait to query the lock after an operation to make sure its not jammed
POST_OPERATION_SYNC_TIME = 10.00

# How long to wait if we get an update storm from the lock
UPDATE_IN_PROGRESS_DEFER_SECONDS = DISCONNECT_DELAY - 1

RETRY_BACKOFF_EXCEPTIONS = (BleakDBusError, DisconnectedError)

RETRY_EXCEPTIONS = (ResponseError, *BLEAK_RETRY_EXCEPTIONS)

# 255 seems to be broadcast randomly when
# there is no update from the lock.
VALID_ADV_VALUES = {0, 1}

AUTH_FAILURE_TO_START_REAUTH = 5

# How long to wait before retrying battery after a timeout (5 minutes)
BATTERY_TIMEOUT_COOLDOWN = 300

# With BATTERY_TIMEOUT_COOLDOWN it may be possible to remove these
# exclusions
NO_BATTERY_SUPPORT_MODELS = {
    "SL-103",  # Linus L2
    "CERES",  # Smart code handle
    "Yale Linus L2",  # Linus L2 Nordic
}

AUTO_LOCK_DEFAULT_DURATION = 90


def operation_lock(func: WrapFuncType) -> WrapFuncType:
    """Define a wrapper to only allow a single operation at a time."""

    async def _async_wrap_operation_lock(
        self: PushLock, *args: Any, **kwargs: Any
    ) -> None:
        _LOGGER.debug("%s: Acquiring lock", self.name)
        async with self._operation_lock:
            return await func(self, *args, **kwargs)

    return cast(WrapFuncType, _async_wrap_operation_lock)


class AuthFailureHistory:
    """Track the number of auth failures."""

    def __init__(self) -> None:
        """Init the history."""
        self._failures_by_mac: dict[str, int] = LRU(1024)

    def auth_failed(self, mac: str) -> None:
        """Increment the number of auth failures."""
        self._failures_by_mac[mac] = self._failures_by_mac.get(mac, 0) + 1

    def auth_success(self, mac: str) -> None:
        """Reset the number of auth failures."""
        self._failures_by_mac[mac] = 0

    def should_raise(self, mac: str) -> bool:
        """Return if we should raise an error."""
        return self._failures_by_mac.get(mac, 0) >= AUTH_FAILURE_TO_START_REAUTH


_AUTH_FAILURE_HISTORY = AuthFailureHistory()


def retry_bluetooth_connection_error(func: WrapFuncType) -> WrapFuncType:
    """
    Define a wrapper to retry on bleak error.

    The accessory is allowed to disconnect us any time so
    we need to retry the operation.
    """

    async def _async_wrap_retry_bluetooth_connection_error(
        self: PushLock, *args: Any, **kwargs: Any
    ) -> Any:
        _LOGGER.debug("%s: Starting retry loop", self.name)
        attempts = DEFAULT_ATTEMPTS
        max_attempts = attempts - 1

        for attempt in range(attempts):
            try:
                return await func(self, *args, **kwargs)
            except AuthError:
                _AUTH_FAILURE_HISTORY.auth_failed(self.address)
                if _AUTH_FAILURE_HISTORY.should_raise(self.address):
                    # If the bluetooth connection drops in the middle of authentication
                    # we may see it as a failed authentication. If we see 5 failed
                    # authentications in a row we can reasonably assume that the key has
                    # changed and we should re-authenticate.
                    self._update_any_state([AuthState(successful=False)])
                    raise
                _LOGGER.debug(
                    "%s: Auth error calling %s, retrying (%s/%s)...",
                    self.name,
                    func,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(0.25)
            except BleakNotFoundError:
                # The lock cannot be found so there is no
                # point in retrying.
                raise
            except RETRY_BACKOFF_EXCEPTIONS as err:
                await self._async_handle_disconnected(err)
                if attempt >= max_attempts:
                    _LOGGER.debug(
                        "%s: %s error calling %s, reach max attempts (%s/%s)",
                        self.name,
                        type(err),
                        func,
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    if is_disconnected_error(err):
                        raise DisconnectedError(str(err)) from err
                    raise
                _LOGGER.debug(
                    "%s: %s error calling %s, backing off %ss, retrying (%s/%s)...",
                    self.name,
                    type(err),
                    func,
                    0.25,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(0.25)
            except RETRY_EXCEPTIONS as err:
                await self._async_handle_disconnected(err)
                if attempt >= max_attempts:
                    _LOGGER.debug(
                        "%s: %s error calling %s, reach max attempts (%s/%s)",
                        self.name,
                        type(err),
                        func,
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    if is_disconnected_error(err):
                        raise DisconnectedError(str(err)) from err
                    raise
                _LOGGER.debug(
                    "%s: %s error calling %s, retrying  (%s/%s)...",
                    self.name,
                    type(err),
                    func,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
        return None

    return cast(WrapFuncType, _async_wrap_retry_bluetooth_connection_error)


class PushLock:
    """A lock with push updates."""

    def __init__(
        self,
        local_name: str | None = None,
        address: str | None = None,
        ble_device: BLEDevice | None = None,
        key: str | None = None,
        key_index: int | None = None,
        advertisement_data: AdvertisementData | None = None,
        idle_disconnect_delay: float = DISCONNECT_DELAY,
        always_connected: bool = False,
        idle_disconnect_delay_pending_update: float = DISCONNECT_DELAY_PENDING_UPDATE,
    ) -> None:
        """Init the lock watcher."""
        if local_name is None and address is None:
            raise ValueError("Must specify either local_name or address")
        if not address and not local_name_is_unique(local_name):
            raise ValueError("local_name must be unique when address is not provided")

        self._local_name = local_name
        self._local_name_is_unique = local_name_is_unique(local_name)
        self._address = address
        self._name: str | None = None
        self._lock_info: LockInfo | None = None
        self._lock_state: LockState | None = None
        self._last_adv_value = -1
        self._last_hk_state = -1
        self._lock_key = key
        self._lock_key_index = key_index
        self._advertisement_data = advertisement_data
        self._ble_device = ble_device
        self._operation_lock = asyncio.Lock()
        self._running = False
        self._callbacks: list[
            Callable[[LockState, LockInfo, ConnectionInfo], None]
        ] = []
        self._update_task: asyncio.Task[None] | None = None
        self.loop = asyncio._get_running_loop()
        self._cancel_deferred_update: asyncio.TimerHandle | None = None
        self._client: Lock | None = None
        self._connect_lock = asyncio.Lock()
        self._seen_this_session: set[
            type[LockStatus | DoorStatus | BatteryState | AuthState | AutoLockState]
        ] = set()
        self._disconnect_timer: asyncio.TimerHandle | None = None
        self._keep_alive_timer: asyncio.TimerHandle | None = None
        self._idle_disconnect_delay_pending_update = (
            idle_disconnect_delay_pending_update
        )
        self._idle_disconnect_delay = idle_disconnect_delay
        self._next_disconnect_delay = idle_disconnect_delay
        self._first_update_future: asyncio.Future[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._last_lock_operation_complete_time = NEVER_TIME
        self._last_operation_complete_time = NEVER_TIME
        self._always_connected = always_connected
        self._next_battery_attempt_time = NEVER_TIME  # Cooldown after battery timeout

    @property
    def local_name(self) -> str | None:
        """Get the local name."""
        return self._local_name

    @property
    def name(self) -> str:
        """Get the name of the lock."""
        if self._name:
            return self._name
        if self._local_name_is_unique and self._local_name:
            return self._local_name
        return self.address

    @property
    def address(self) -> str:
        """Get the address of the lock."""
        if self._ble_device:
            return self._ble_device.address
        assert self._address is not None  # nosec
        return self._address

    @property
    def door_status(self) -> DoorStatus:
        """Return the current door status."""
        return self._lock_state.door if self._lock_state else DoorStatus.UNKNOWN

    @property
    def lock_status(self) -> LockStatus:
        """Return the current lock status."""
        return self._lock_state.lock if self._lock_state else LockStatus.UNKNOWN

    @property
    def battery(self) -> BatteryState | None:
        """Return the current battery state."""
        return self._lock_state.battery if self._lock_state else None

    @property
    def auth(self) -> AuthState | None:
        """Return the current auth state."""
        return self._lock_state.auth if self._lock_state else None

    @property
    def auto_lock(self) -> AutoLockState | None:
        """Return the current auto lock state."""
        return self._lock_state.auto_lock if self._lock_state else None

    @property
    def auto_lock_prev(self) -> AutoLockState | None:
        """Return the previous auto lock state."""
        return self._lock_state.auto_lock_prev if self._lock_state else None

    @property
    def lock_state(self) -> LockState | None:
        """Return the current lock state."""
        return self._lock_state

    @property
    def lock_info(self) -> LockInfo | None:
        """Return the current lock info."""
        return self._lock_info

    @property
    def connection_info(self) -> ConnectionInfo | None:
        """Return the current connection info."""
        if self._advertisement_data:
            return ConnectionInfo(self._advertisement_data.rssi)
        return None

    @property
    def ble_device(self) -> BLEDevice | None:
        """Return the current BLEDevice."""
        return self._ble_device

    @property
    def is_connected(self) -> bool:
        """Return if the lock is connected."""
        return bool(self._client and self._client.is_connected)

    def set_name(self, name: str) -> None:
        """Set the name of the lock."""
        self._name = name

    def reset_advertisement_state(self) -> None:
        """Reset the advertisement state."""
        self._last_adv_value = -1
        self._last_hk_state = -1

    def register_callback(
        self, callback: Callable[[LockState, LockInfo, ConnectionInfo], None]
    ) -> Callable[[], None]:
        """Register a callback to be called when the lock state changes."""

        def unregister_callback() -> None:
            self._callbacks.remove(callback)

        self._callbacks.append(callback)
        return unregister_callback

    def set_lock_key(self, key: str, slot: int) -> None:
        """Set the lock key."""
        self._lock_key = key
        self._lock_key_index = slot

    def set_ble_device(self, ble_device: BLEDevice) -> None:
        """Set the ble device."""
        self._ble_device = ble_device
        self._address = ble_device.address

    def set_advertisement_data(self, advertisement_data: AdvertisementData) -> None:
        """Set the advertisement data."""
        self._advertisement_data = advertisement_data

    def _get_lock_instance(self) -> Lock:
        """Get the lock instance."""
        assert self._ble_device is not None  # nosec
        assert self._lock_key is not None  # nosec
        assert self._lock_key_index is not None  # nosec
        return Lock(
            lambda: self._ble_device,
            self._lock_key,
            self._lock_key_index,
            self.name,
            self._state_callback,
            self._lock_info,
            self._disconnected_callback,
        )

    def _disconnected_callback(self) -> None:
        """Handle a disconnect from the lock."""
        _LOGGER.debug("%s: Disconnected from lock callback", self.name)
        if self._always_connected and not _AUTH_FAILURE_HISTORY.should_raise(
            self.address
        ):
            _LOGGER.debug(
                "%s: Scheduling reconnect from disconnected callback", self.name
            )
            self._keep_alive()

    def _keep_alive(self) -> None:
        """Keep the lock connection alive."""
        if not self._always_connected:
            return
        _LOGGER.debug("%s: Executing keep alive", self.name)
        self._schedule_future_update(0)
        self._schedule_next_keep_alive(KEEP_ALIVE_TIME)

    def _time_since_last_operation(self) -> float:
        """Return the time since the last operation."""
        return time.monotonic() - self._last_operation_complete_time

    def _reschedule_next_keep_alive(self) -> None:
        """Reschedule the next keep alive."""
        next_keep_alive_time = max(
            0, KEEP_ALIVE_TIME - self._time_since_last_operation()
        )
        self._schedule_next_keep_alive(next_keep_alive_time)

    def _schedule_next_keep_alive(self, delay: float) -> None:
        """Schedule the next keep alive."""
        self._cancel_keepalive_timer()
        if not self._always_connected or not self._running:
            return
        _LOGGER.debug(
            "%s: Scheduling next keep alive in %s seconds",
            self.name,
            delay,
        )
        self._keep_alive_timer = self.loop.call_later(
            delay,
            self._keep_alive,
        )

    def _reset_disconnect_timer(self) -> None:
        """Reset disconnect timer."""
        if self._always_connected and self._running:
            return
        self._cancel_disconnect_timer()
        self._expected_disconnect = False
        timeout = self._next_disconnect_delay
        _LOGGER.debug(
            "%s: Resetting disconnect timer to %s seconds", self.name, timeout
        )
        self._disconnect_timer = self.loop.call_later(
            timeout, self._disconnect_with_timer, timeout
        )

    async def _execute_forced_disconnect(self, reason: str) -> None:
        """Execute forced disconnection."""
        self._cancel_disconnect_timer()
        _LOGGER.debug("%s: Executing forced disconnect: %s", self.name, reason)
        if (update_task := self._update_task) and not update_task.done():
            self._update_task = None
            update_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await update_task
        await self._execute_disconnect()

    def _disconnect_with_timer(self, timeout: float) -> None:
        """
        Disconnect from device.

        This should only ever be called from _reset_disconnect_timer
        """
        if self._operation_lock.locked():
            _LOGGER.debug("%s: Disconnect timer reset due to operation lock", self.name)
            self._reset_disconnect_timer()
            return
        if self._cancel_deferred_update:
            _LOGGER.debug(
                "%s: Disconnect timer fired while we were waiting to update", self.name
            )
            self._reset_disconnect_timer()
            self._cancel_future_update()
            self._deferred_update()
            return
        self._cancel_disconnect_timer()
        self.background_task(self._execute_timed_disconnect(timeout))

    def _cancel_disconnect_timer(self) -> None:
        """Cancel disconnect timer."""
        if self._disconnect_timer:
            self._disconnect_timer.cancel()
            self._disconnect_timer = None

    def _cancel_keepalive_timer(self) -> None:
        """Cancel keep alive timer."""
        if self._keep_alive_timer:
            self._keep_alive_timer.cancel()
            self._keep_alive_timer = None

    async def _execute_timed_disconnect(self, timeout: float) -> None:
        """Execute timed disconnection."""
        _LOGGER.debug(
            "%s: Executing timed disconnect after timeout of %s",
            self.name,
            timeout,
        )
        await self._execute_disconnect()

    async def _async_handle_disconnected(self, exc: Exception) -> None:
        """Clean up after a disconnect."""
        _LOGGER.debug("%s: Disconnected due to %s, cleaning up", self.name, exc)
        if self._connect_lock.locked():
            _LOGGER.error(
                "%s: Disconnected while connection was in progress, ignoring",
                self.name,
            )
            return
        self._cancel_disconnect_timer()
        await self._execute_disconnect()

    async def _execute_disconnect(self) -> None:
        """Execute disconnection."""
        async with self._connect_lock:
            if (
                self._running and self._disconnect_timer
            ):  # If the timer was reset, don't disconnect
                return
            client = self._client
            self._client = None
            if client:
                _LOGGER.debug("%s: Disconnecting", self.name)
                await client.disconnect()
                _LOGGER.debug("%s: Disconnect completed", self.name)

    async def _ensure_connected(self) -> Lock:
        """Ensure connection to device is established."""
        if self._connect_lock.locked():
            self._reset_disconnect_timer()
            _LOGGER.debug(
                "%s: Connection already in progress, waiting for it to complete",
                self.name,
            )
        if self.is_connected:
            assert self._client is not None  # nosec
            self._reset_disconnect_timer()
            return self._client
        async with self._connect_lock:
            # Check again while holding the lock
            if self.is_connected:
                assert self._client is not None  # type: ignore[unreachable] # nosec
                self._reset_disconnect_timer()
                return self._client
            self._client = self._get_lock_instance()
            max_attempts = 1 if self._first_update_future else MAX_CONNECT_ATTEMPTS
            try:
                await self._client.connect(max_attempts)
            except BaseException as ex:  # Might be cancelled
                _LOGGER.debug(
                    "%s: Failed to connect due to %s, forcing disconnect", self.name, ex
                )
                try:
                    await self._client.disconnect()
                except Exception:
                    _LOGGER.exception(
                        "%s: Failed to disconnect after failed connect", self.name
                    )
                raise
            self._next_disconnect_delay = self._idle_disconnect_delay
            self._reset_disconnect_timer()
            self._seen_this_session.clear()
            return self._client

    async def securemode(self) -> None:
        """Set the lock into securemode."""
        self._update_any_state([LockStatus.LOCKING])
        self._cancel_future_update()
        await self._execute_lock_operation(
            "force_securemode", LockStatus.LOCKING, LockStatus.SECUREMODE
        )

    async def lock(self) -> None:
        """Lock the lock."""
        self._update_any_state([LockStatus.LOCKING])
        self._cancel_future_update()
        await self._execute_lock_operation(
            "force_lock", LockStatus.LOCKING, LockStatus.LOCKED
        )

    async def unlock(self) -> None:
        """Unlock the lock."""
        self._update_any_state([LockStatus.UNLOCKING])
        self._cancel_future_update()
        await self._execute_lock_operation(
            "force_unlock", LockStatus.UNLOCKING, LockStatus.UNLOCKED
        )

    @operation_lock
    @retry_bluetooth_connection_error
    async def _execute_lock_operation(
        self, op_attr: str, pending_state: LockStatus, complete_state: LockStatus
    ) -> None:
        """Execute a lock operation."""
        if not self._running:
            raise RuntimeError(
                f"{self.name}: Lock operation not possible because not running"
            )
        _LOGGER.debug("%s: Starting %s", self.name, pending_state)
        self._update_any_state([pending_state])
        self._cancel_future_update()
        try:
            lock = await self._ensure_connected()
            self._cancel_future_update()
            await getattr(lock, op_attr)()
        except Exception as ex:
            self._update_any_state([LockStatus.UNKNOWN])
            _LOGGER.debug(
                "%s: Failed to execute lock operation due to %s, forcing disconnect",
                self.name,
                ex,
            )
            raise
        self._update_any_state([complete_state])
        _LOGGER.debug("%s: Finished %s", self.name, complete_state)
        now = time.monotonic()
        self._last_lock_operation_complete_time = now
        self._complete_operation(now)

    @property
    def auto_lock_durations(self) -> list[int]:
        return [0, 10, 30, 60, 90, 120, 150, 180, 240, 300, 600, 1200, 1800]

    @property
    def auto_lock_modes(self) -> list[str]:
        return ["off", "instant", "timer"]

    async def set_auto_lock_mode(self, mode: AutoLockMode) -> None:
        """Set auto lock setting."""
        if mode == AutoLockMode.OFF:
            if self.auto_lock and self.auto_lock.mode == AutoLockMode.OFF:
                _LOGGER.debug("%s: Auto lock is already off", self.name)
                return
            await self._set_auto_lock(AutoLockMode.OFF, 0)
            return

        duration = AUTO_LOCK_DEFAULT_DURATION
        if self.auto_lock and self.auto_lock.mode != AutoLockMode.OFF:
            duration = self.auto_lock.duration
        elif self.auto_lock_prev and self.auto_lock_prev.mode != AutoLockMode.OFF:
            # If the auto lock is currently off, use the previous duration
            duration = self.auto_lock_prev.duration
        await self._set_auto_lock(mode, duration)

    async def set_auto_lock_duration(self, duration: int) -> None:
        """Set auto lock setting."""
        if duration == 0:
            if self.auto_lock and self.auto_lock.mode == AutoLockMode.OFF:
                _LOGGER.debug("%s: Auto lock is already off", self.name)
                return
            await self._set_auto_lock(AutoLockMode.OFF, 0)
            return

        mode = AutoLockMode.TIMER
        if self.auto_lock and self.auto_lock.mode != AutoLockMode.OFF:
            mode = self.auto_lock.mode
        elif self.auto_lock_prev and self.auto_lock_prev.mode != AutoLockMode.OFF:
            # If the auto lock is currently off, use the previous mode
            mode = self.auto_lock_prev.mode
        await self._set_auto_lock(mode, duration)

    @retry_bluetooth_connection_error
    async def _set_auto_lock(self, mode: AutoLockMode, duration: int) -> None:
        """Set auto lock setting."""
        if not self._running:
            raise RuntimeError(
                f"{self.name}: Set auto lock operation not possible because not running"
            )
        # Duration validation
        if duration not in self.auto_lock_durations:
            raise ValueError(f"Invalid auto lock duration: {duration}")
        try:
            lock = await self._ensure_connected()
            self._cancel_future_update()
            await lock.set_auto_lock(mode, duration)
        except Exception as ex:
            _LOGGER.debug(
                "%s: Failed to execute set auto lock operation due to %s, "
                "forcing disconnect",
                self.name,
                ex,
            )
            raise
        self._complete_operation(time.monotonic())

    def _complete_operation(self, now: float) -> None:
        """Mark an operation as complete and reset timers."""
        self._last_operation_complete_time = now
        self._reset_disconnect_timer()
        self._reschedule_next_keep_alive()

    def _state_callback(self, states: Iterable[LockStateValue]) -> None:
        """Handle state change."""
        self._reset_disconnect_timer()
        self._update_any_state(states)

    def _get_current_state(self) -> LockState:
        """Get the current state of the lock."""
        return self._lock_state or LockState(
            self.lock_status,
            self.door_status,
            self.battery,
            self.auth,
            self.auto_lock,
            self.auto_lock_prev,
        )

    def _update_any_state(self, states: Iterable[LockStateValue | AuthState]) -> None:
        _LOGGER.debug("%s: State changed: %s", self.name, states)
        lock_state = self._get_current_state()
        original_lock_status = lock_state.lock
        changes: dict[str, Any] = {}
        for state in states:
            state_type = type(state)
            self._seen_this_session.add(state_type)
            if isinstance(state, AuthState):
                if lock_state.auth != state:
                    changes["auth"] = state
            elif isinstance(state, LockStatus):
                if lock_state.lock != state:
                    changes["lock"] = state
            elif isinstance(state, DoorStatus):
                if lock_state.door != state:
                    changes["door"] = state
            elif isinstance(state, BatteryState):
                if state.voltage <= 3.0:
                    _LOGGER.debug(
                        "%s: Battery voltage is impossible: %s",
                        self.name,
                        state.voltage,
                    )
                    continue
                if lock_state.battery != state:
                    changes["battery"] = state
            elif isinstance(state, AutoLockState):
                if lock_state.auto_lock != state:
                    changes["auto_lock"] = state
                    changes["auto_lock_prev"] = lock_state.auto_lock
            else:
                raise ValueError(f"Unexpected state type: {state}")

        if not changes:
            return

        lock_state = replace(lock_state, **changes)
        if (
            original_lock_status != lock_state.lock
            and (not lock_state.auth or lock_state.auth.successful)
            and original_lock_status != LockStatus.UNKNOWN
        ):
            self._schedule_future_update(RESYNC_DELAY)

        self._callback_state(lock_state)

    async def update(self) -> None:
        """Request that status be updated."""
        _LOGGER.debug("%s: Starting manual update", self.name)
        self._schedule_future_update_with_debounce(
            0 if self.is_connected else MANUAL_UPDATE_COALESCE_SECONDS
        )

    async def validate(self) -> None:
        """Validate lock credentials."""
        _LOGGER.debug("%s: Starting validate", self.name)
        await self._update()
        _LOGGER.debug("%s: Finished validate", self.name)

    async def _poll_battery(
        self, lock: Lock, state: LockState
    ) -> tuple[LockState, bool]:
        """Poll battery status with timeout handling and cooldown.

        Returns tuple of (updated_state, made_request).
        """
        # Check battery timeout cooldown
        now = time.monotonic()
        battery_on_cooldown = now < self._next_battery_attempt_time

        if battery_on_cooldown:
            _LOGGER.debug(
                "%s: Skipping battery request due to recent timeout "
                "(cooldown until %.1fs)",
                self.name,
                self._next_battery_attempt_time - now,
            )
            return state, False

        try:
            battery_state = await lock.battery()
            _AUTH_FAILURE_HISTORY.auth_success(self.address)
            state = replace(
                state, battery=battery_state, auth=AuthState(successful=True)
            )
            # Reset cooldown on success
            self._next_battery_attempt_time = NEVER_TIME
        except TimeoutError as err:
            _LOGGER.info(
                "%s: Battery request timed out (%s), will retry in %d "
                "seconds. Continuing with other updates.",
                self.name,
                err,
                BATTERY_TIMEOUT_COOLDOWN,
            )
            # Set cooldown to prevent repeated timeouts
            self._next_battery_attempt_time = now + BATTERY_TIMEOUT_COOLDOWN
        except (BleakError, BleakDBusError) as err:
            _LOGGER.debug(
                "%s: Battery request failed (%s), continuing with other updates.",
                self.name,
                err,
            )

        return state, True

    @operation_lock
    @retry_bluetooth_connection_error
    async def _update(self) -> LockState:
        """Update the lock state."""
        has_lock_info = self._lock_info is not None

        _LOGGER.debug(
            "%s: Starting update (has_lock_info: %s)", self.name, has_lock_info
        )
        lock = await self._ensure_connected()
        if not self._lock_info:
            self._lock_info = await lock.lock_info()
            _LOGGER.debug("Obtained lock info: %s", self._lock_info)
        # Asking for battery first seems to be reduce the chance of the lock
        # getting into a bad state.
        state = self._get_current_state()
        made_request = False

        needs_battery_workaround = self._lock_info.model in NO_BATTERY_SUPPORT_MODELS
        _LOGGER.debug(
            "Needs battery workaround model %s: %s",
            self._lock_info.model,
            needs_battery_workaround,
        )

        if not needs_battery_workaround and BatteryState not in self._seen_this_session:
            state, battery_requested = await self._poll_battery(lock, state)
            if battery_requested:
                made_request = True

        if (
            DoorStatus not in self._seen_this_session
            and self._lock_info
            and self._lock_info.door_sense
        ):
            made_request = True
            door_status = await lock.door_status()
            _AUTH_FAILURE_HISTORY.auth_success(self.address)
            state = replace(state, door=door_status, auth=AuthState(successful=True))

        if AutoLockState not in self._seen_this_session:
            made_request = True
            auto_lock_state = await lock.auto_lock_status()
            _AUTH_FAILURE_HISTORY.auth_success(self.address)
            state = replace(
                state,
                auto_lock=auto_lock_state,
                auto_lock_prev=state.auto_lock,
                auth=AuthState(successful=True),
            )

        # Only ask for the lock status if we haven't seen
        # it this session since notify callbacks will happen
        # if it changes and the extra polling can cause the lock
        # to get into a bad state.
        #
        # However, we always want to poll lock
        # state to keep the connection alive if we are always connected.
        if LockStatus not in self._seen_this_session or (
            not made_request and self._always_connected
        ):
            made_request = True
            lock_status = await lock.lock_status()
            _AUTH_FAILURE_HISTORY.auth_success(self.address)
            state = replace(state, lock=lock_status, auth=AuthState(successful=True))

        _LOGGER.debug("%s: Finished update", self.name)

        self._callback_state(state)

        if state.battery and state.battery.voltage <= 3.0:
            _LOGGER.debug(
                "%s: Battery voltage is impossible: %s",
                self.name,
                state.battery.voltage,
            )
            # If the battery voltage is impossible, reconnect.
            await self._execute_forced_disconnect("impossible battery voltage")

        if state.lock in (LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06):
            _LOGGER.debug("%s: Lock is in an unknown state: %s", self.name, state.lock)
            # If the lock is in a bad state, reconnect.
            await self._execute_forced_disconnect(
                f"lock is in unknown state: {state.lock}"
            )

        if not has_lock_info:
            # On first update free up the connection
            # so we can bring other locks online if
            # the bluetooth adapter is out of connections
            # slots. We reset the timer to a low number
            # so that if another update request is pending
            # we do not disconnect until it completes.
            self._next_disconnect_delay = FIRST_CONNECTION_DISCONNECT_TIME
            self._reset_disconnect_timer()

        if made_request:
            self._last_operation_complete_time = time.monotonic()
            self._reschedule_next_keep_alive()
        return state

    def _callback_state(self, lock_state: LockState) -> None:
        """Call the callbacks."""
        self._lock_state = lock_state
        _LOGGER.debug(
            "%s: New state: %s %s %s",
            self.name,
            self._lock_state,
            self._lock_info,
            self.connection_info,
        )
        if not self._callbacks:
            return
        assert self._lock_info is not None  # nosec
        connection_info = self.connection_info
        assert connection_info is not None  # nosec
        for callback in self._callbacks:
            try:
                callback(lock_state, self._lock_info, connection_info)
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("%s: Error calling callback", self.name)

    def update_advertisement(
        self, ble_device: BLEDevice, ad: AdvertisementData
    ) -> None:
        """Update the advertisement."""
        adv_debug_enabled = _ADV_LOGGER.isEnabledFor(logging.DEBUG)
        if self._local_name_is_unique and self._local_name == ad.local_name:
            if adv_debug_enabled:
                _ADV_LOGGER.debug(
                    "%s: Accepting new advertisement since local_name %s matches: %s",
                    self.name,
                    ad.local_name,
                    ad,
                )
        elif self.address and self.address == ble_device.address:
            if adv_debug_enabled:
                _ADV_LOGGER.debug(
                    "%s: Accepting new advertisement since address %s matches: %s",
                    self.name,
                    self.address,
                    ad,
                )
        else:
            return
        self.set_ble_device(ble_device)
        self.set_advertisement_data(ad)
        next_update = 0.0
        mfr_data = ad.manufacturer_data
        if APPLE_MFR_ID in mfr_data:
            first_byte = mfr_data[APPLE_MFR_ID][0]
            if first_byte == HAP_FIRST_BYTE:
                hk_state = get_homekit_state_num(mfr_data[APPLE_MFR_ID])
                # Sometimes the yale data is glued on to the end of the HomeKit data
                # but in that case it seems wrong so we don't process it
                #
                # if len(mfr_data[APPLE_MFR_ID]) > 20 and YALE_MFR_ID not in mfr_data:
                # mfr_data[YALE_MFR_ID] = mfr_data[APPLE_MFR_ID][20:]
                if self._last_hk_state == -1:
                    # We haven't seen a HomeKit state yet so we schedule an update
                    next_update = FIRST_UPDATE_COALESCE_SECONDS
                elif hk_state != self._last_hk_state:
                    next_update = HK_UPDATE_COALESCE_SECONDS
                self._last_hk_state = hk_state
            elif first_byte == HAP_ENCRYPTED_FIRST_BYTE:
                # Encrypted data, we don't know how to decrypt it
                # but we know its a state change so we schedule an update
                next_update = HK_UPDATE_COALESCE_SECONDS
        # Only track the single 0/1 value from the advertisement
        # as we can get an storm of metadata we don't know how to
        # decode that starts with b'\x00\x00' and will cause us to
        # connect over and over again when active scanning is enabled.
        # b'\x00\x00\x80\x15\xd0\x11\xf7\xa5\x43\x1f\x85\xd7\xff\x23\x5f\x1e\x75\x46'
        is_first_advertisement = self._last_adv_value == -1
        if YALE_MFR_ID in mfr_data and (
            len(mfr_data[YALE_MFR_ID]) == 1 or is_first_advertisement
        ):
            current_value = mfr_data[YALE_MFR_ID][0]
            if not next_update:
                if is_first_advertisement:
                    # We haven't seen a valid value yet so we schedule an update
                    next_update = FIRST_UPDATE_COALESCE_SECONDS
                elif (
                    current_value in VALID_ADV_VALUES
                    and current_value != self._last_adv_value
                ):
                    next_update = ADV_UPDATE_COALESCE_SECONDS
            self._last_adv_value = current_value
        if adv_debug_enabled:
            scheduled_update = None
            if self._cancel_deferred_update:
                scheduled_update = (
                    self._cancel_deferred_update.when() - self.loop.time()
                )
            _ADV_LOGGER.debug(
                "%s: State: (current_state: %s) (hk_state: %s) "
                "(adv_value: %s) (next_update: %s) (scheduled_update: %s)",
                self.name,
                self._lock_state,
                self._last_hk_state,
                self._last_adv_value,
                next_update,
                scheduled_update,
            )
        if not next_update:
            return
        if (
            self.is_connected
            and self._next_disconnect_delay != FIRST_CONNECTION_DISCONNECT_TIME
            and (
                self._time_since_last_operation()
                + self._idle_disconnect_delay_pending_update
            )
            < KEEP_ALIVE_TIME
        ):
            # Already connected, state will be pushed, but stay
            # connected a bit longer to make sure we get it unless
            # this is the first connection or deferring the update
            # would keep the connection idle for too long and
            # get us disconnected anyways.
            self._next_disconnect_delay = self._idle_disconnect_delay_pending_update
            self._reset_disconnect_timer()
            return
        self._schedule_future_update_with_debounce(next_update)

    async def start(self) -> Callable[[], None]:
        """Start watching for updates."""
        _LOGGER.debug("Waiting for advertisement callbacks for %s", self.name)
        if self._running:
            raise RuntimeError("Already running")
        self._running = True
        self._first_update_future = asyncio.get_running_loop().create_future()
        if device := await get_device(self.address):
            self.set_ble_device(device)
            self._schedule_future_update_with_debounce(ADV_UPDATE_COALESCE_SECONDS)

        return self._cancel

    def _cancel(self) -> None:
        self._running = False
        self._cancel_future_update()
        self.background_task(self._execute_forced_disconnect("stopping"))

    def background_task(self, fut: Coroutine[Any, Any, Any]) -> None:
        """Execute a background task."""
        task: asyncio.Task[Any] = asyncio.create_task(fut)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.remove)

    async def wait_for_first_update(self, timeout: float) -> None:
        """Wait for the first update."""
        if not self._running:
            raise RuntimeError("Not running")
        if not self._first_update_future:
            raise RuntimeError("Already waited for first update")
        try:
            async with asyncio_timeout(timeout):
                await self._first_update_future
        except (TimeoutError, asyncio.CancelledError) as ex:
            self._first_update_future.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._first_update_future
            raise NoAdvertisementError(
                "No advertisement received before timeout"
            ) from ex
        finally:
            self._first_update_future = None

    def _cancel_future_update(self) -> None:
        """Cancel an update."""
        if self._cancel_deferred_update:
            self._cancel_deferred_update.cancel()
            self._cancel_deferred_update = None

    def _schedule_future_update_with_debounce(self, seconds: float) -> None:
        """Schedule an update with a potential debounce."""
        future_update_time = seconds
        if self._cancel_deferred_update:
            time_till_update = self._cancel_deferred_update.when() - self.loop.time()
            if time_till_update < HK_UPDATE_COALESCE_SECONDS:
                future_update_time = HK_UPDATE_COALESCE_SECONDS
                _LOGGER.debug(
                    "%s: Existing update too soon %s, "
                    "rescheduling update for in %s seconds",
                    self.name,
                    time_till_update,
                    future_update_time,
                )
            elif time_till_update < seconds:
                _LOGGER.debug(
                    "%s: Existing update in %s seconds will happen sooner than now",
                    self.name,
                    time_till_update,
                )
                return
            _LOGGER.debug(
                "%s: Rescheduling update for %s", self.name, future_update_time
            )
        self._schedule_future_update(future_update_time)

    def _schedule_future_update(self, future_update_time: float) -> None:
        """Schedule an update in future seconds."""
        _LOGGER.debug(
            "%s: Scheduling update to happen in %s seconds",
            self.name,
            future_update_time,
        )
        self._cancel_future_update()
        self._cancel_deferred_update = self.loop.call_later(
            future_update_time, self._deferred_update
        )

    def _deferred_update(self) -> None:
        """Update the lock state."""
        self._cancel_future_update()
        now = time.monotonic()
        if self._update_task and not self._update_task.done():
            _LOGGER.debug(
                "%s: Rescheduling update since one already in progress", self.name
            )
            self._schedule_future_update_with_debounce(UPDATE_IN_PROGRESS_DEFER_SECONDS)
            return
        if (
            seconds_time_lock_op := (now - self._last_lock_operation_complete_time)
        ) < LOCK_STALE_STATE_DEBOUNCE_DELAY:
            _LOGGER.debug("%s: Rescheduling update to avoid stale state", self.name)
            self._schedule_future_update_with_debounce(seconds_time_lock_op)
            return
        self._update_task = asyncio.create_task(self._execute_deferred_update())

    def _set_update_state(self, exception: Exception | None) -> None:
        """Set the update state."""
        if not self._first_update_future:
            return
        if exception:
            self._first_update_future.set_exception(exception)
        else:
            self._first_update_future.set_result(None)

    async def _execute_deferred_update(self) -> None:
        """Execute deferred update."""
        _LOGGER.debug("%s: Deferred update starting", self.name)
        if not self._running:
            _LOGGER.debug("%s: Deferred updated ignored because not running", self.name)
            return
        _LOGGER.debug("%s: Starting deferred update", self.name)
        try:
            await self._update()
            self._set_update_state(None)
        except AuthError as ex:
            self._set_update_state(ex)
            _LOGGER.exception(
                "%s: Auth error: key or slot (key index) is incorrect",
                self.name,
            )
        except asyncio.CancelledError:
            self._set_update_state(RuntimeError("Update was canceled"))
            _LOGGER.debug("%s: In-progress update canceled", self.name)
            raise
        except TimeoutError as ex:
            self._set_update_state(ex)
            _LOGGER.exception("%s: Timed out updating", self.name)
        except BleakNotFoundError as ex:
            wrapped_bleak_exc = BluetoothError(str(ex))
            wrapped_bleak_exc.__cause__ = ex
            self._set_update_state(wrapped_bleak_exc)
            _LOGGER.debug("%s: not found error updating", self.name, exc_info=True)
        except BleakError as ex:
            wrapped_bleak_exc = BluetoothError(str(ex))
            wrapped_bleak_exc.__cause__ = ex
            self._set_update_state(wrapped_bleak_exc)
            _LOGGER.exception("%s: Bluetooth error updating", self.name)
        except DisconnectedError as ex:
            wrapped_bleak_exc = BluetoothError(str(ex))
            wrapped_bleak_exc.__cause__ = ex
            self._set_update_state(wrapped_bleak_exc)
            _LOGGER.exception("%s: Disconnected while updating", self.name)
        except Exception as ex:  # pylint: disable=broad-except
            wrapped_exc = YaleXSBLEError(str(ex))
            wrapped_exc.__cause__ = ex
            self._set_update_state(wrapped_exc)
            _LOGGER.exception("%s: Unknown error updating", self.name)


def get_homekit_state_num(data: bytes) -> int:
    """Get the homekit state number from the manufacturer data."""
    _acid, gsn, _cn, _cv = struct.unpack("<HHBB", data[9:15])
    return gsn
