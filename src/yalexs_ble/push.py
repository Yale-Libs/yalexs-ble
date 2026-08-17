from __future__ import annotations

import asyncio
import contextlib
import functools
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
    OperationFailedError,
    OperationIncompleteError,
    ResponseError,
    UnlatchError,
    YaleXSBLEError,
)
from .util import asyncio_timeout, is_disconnected_error, local_name_is_unique

_LOGGER = logging.getLogger(__name__)

# Advertisement debugger (this one is quite noisy so it has its only logger)
_ADV_LOGGER = logging.getLogger("yalexs_ble_adv")

WrapFuncType = TypeVar("WrapFuncType", bound=Callable[..., Any])

# A monotonic timestamp ~one day in the past, used as a "never happened /
# no deadline" sentinel. Makes no assumption about the clock's epoch.
NEVER_TIME = time.monotonic() - 86400.0

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

# How long to hold a read off the lock from an observed op-response. The
# op-response is the end of the movement itself, so it needs less margin than
# the 6.1 s measured from a command that has not answered yet. It is the hold
# an operation another central ran gets, since nothing else here marks one.
POST_OP_RESPONSE_DEBOUNCE_DELAY = 4.1

# How long to wait before processing an advertisement change
ADV_UPDATE_COALESCE_SECONDS = 0.05

# How long to wait before processing the first update
FIRST_UPDATE_COALESCE_SECONDS = 0.01

# How long to wait before processing a HomeKit advertisement change
HK_UPDATE_COALESCE_SECONDS = 0.025

# How long to wait before processing a manual update request
MANUAL_UPDATE_COALESCE_SECONDS = 0.05

# BLE connection parameters for always-connected mode (battery saving)
# After the initial sync, we switch to a low duty cycle to conserve battery.
# Values are in BLE units: intervals in 1.25ms, timeout in 10ms.
#
# The idle duty cycle is set by peripheral latency, not by the interval: the
# lock may skip up to SLOW_LATENCY connection events, so it wakes about every
# (1 + SLOW_LATENCY) * interval = 510ms. Keeping the interval short means the
# lock drops latency and drains its notifications at the base interval as soon
# as it has something to send. Pinning min == max at a long interval instead
# (1000ms) makes notification delivery, which is acknowledgement gated at one
# frame per two connection events, take ~2s per frame; a lock operation's
# three-frame reply then needs >6s to drain and the next command is issued
# while the previous operation's frames are still arriving.
SLOW_MIN_INTERVAL = 24  # 30ms
SLOW_MAX_INTERVAL = 24  # 30ms
SLOW_LATENCY = 16  # up to 16 skipped connection events (510ms)
SLOW_TIMEOUT = 600  # 6000ms (spec minimum here is (1 + 16) * 30ms * 2 = 1020ms)

# How long to wait to query the lock after an operation to make sure its not jammed
POST_OPERATION_SYNC_TIME = 10.00

# How long a fresh JAMMED status stays pinned on display, as a deadline that
# every admitted jam event pushes out again (now + JAMMED_HOLD_TIME, the
# _poll_battery deadline pattern). Post-jam GETSTATUS fabricates a plain
# position (14/14 field jams; JAMMED was clobbered in under a second), and
# the end of a jam cannot be determined remotely, so the hold's primary
# purpose is making the user aware of the jam via the UI. The hold masks
# only the reported lock status and books no poll while it runs, so it has
# no connection side effect; a wakeup at the hold's end schedules the
# follow-up poll that reads the lock's current state back. 30 s is a
# nominal visibility window.
JAMMED_HOLD_TIME = 30.0

# How long to wait before checking again when an update falls due while an
# operation is running. No update cycle may be created during an operation,
# because the cycle would wait on the operation lock and run the instant the
# operation ends, inside the settle window. The check backs off briefly
# instead and lets the retry find the operation finished or the moment still
# passed.
DEADLINE_WAKEUP_RETRY_DELAY = 1.0

# How long to wait if we get an update storm from the lock
UPDATE_IN_PROGRESS_DEFER_SECONDS = DISCONNECT_DELAY - 1

# Lock statuses that report a position the lock is holding, and holds until an
# operation or a person moves it. UNKNOWN_01 and UNKNOWN_06 belong here because
# calibration and polarity discovery are setup conditions that end at the lock
# by hand, so the reported value stands until someone acts on it.
#
# Every other status is a mechanism still moving, the momentary UNLATCHED the
# lock leaves on its own once the latch returns, or the UNKNOWN a failed
# operation stamps. Holding one of those must not count as having seen the lock
# status this session, because that suppresses the follow-up lock_status() poll
# in _update, and that poll's reading is what replaces the value once the
# mechanism stops.
#
# The positions are the side that is enumerated, so a status this set does not
# name costs a poll rather than leaving the display on a value with nothing
# booked to replace it.
POSITION_READINGS = frozenset(
    {
        LockStatus.LOCKED,
        LockStatus.UNLOCKED,
        LockStatus.SECUREMODE,
        LockStatus.JAMMED,
        LockStatus.UNKNOWN_01,
        LockStatus.UNKNOWN_06,
    }
)

RETRY_BACKOFF_EXCEPTIONS = (BleakDBusError, DisconnectedError)

RETRY_EXCEPTIONS = (ResponseError, *BLEAK_RETRY_EXCEPTIONS)

RETRYABLE_EXCEPTIONS = (*RETRY_BACKOFF_EXCEPTIONS, *RETRY_EXCEPTIONS)

# 255 seems to be broadcast randomly when
# there is no update from the lock.
VALID_ADV_VALUES = {0, 1}

# The HomeKit advertisement fields read below end at byte 15: the global state
# number sits at [11:13] inside the <HHBB record that starts at byte 9. A
# payload shorter than this carries no state number to read, so it is skipped
# rather than unpacked.
HAP_STATE_RECORD_END = 15

AUTH_FAILURE_TO_START_REAUTH = 5

# How long to wait before retrying battery after a timeout (5 minutes)
BATTERY_TIMEOUT_COOLDOWN = 300

# How often to re-poll battery state in always_connected mode (10 minutes)
BATTERY_REFRESH_INTERVAL = 600

# How often to re-read the auto lock setting after a successful read (1 hour).
# Auto lock is configuration state that changes rarely, so an hourly refresh
# catches an out-of-band change while keeping the read off every keep-alive
# cycle to save battery. Mirrors BATTERY_REFRESH_INTERVAL.
AUTO_LOCK_READ_REFRESH_INTERVAL = 3600

# How long to stop reading the auto lock setting after it goes unanswered
# (24 hours). The state is in memory, so a large value means "until restart".
# Longer than BATTERY_TIMEOUT_COOLDOWN because an unanswered read points to a
# lock that does not support the setting, so a long quiet window is wanted.
AUTO_LOCK_READ_FAILURE_BACKOFF = 86400

# How many consecutive unanswered reads before backing off. Three in a row is
# the signal the lock does not support the setting; a success resets the count.
# Ack timeouts and response timeouts both count toward this one threshold.
AUTO_LOCK_READ_FAILURE_THRESHOLD = 3

# How long to wait for the 0xBB settings response after the READSETTING ack
# before treating the read as unresolved. The ack completes the solicited wait;
# the value follows moments later on the notify path. This must clear two bounds:
# above SLOW_TIMEOUT (the 6s slow-connection supervision timeout, 600 in 10ms
# units) so a slow but alive link is not struck before it could deliver, and
# below KEEP_ALIVE_TIME so the next cycle sees it lapsed when the value never
# comes. Mirrors the session command timeout.
AUTO_LOCK_READ_RESPONSE_TIMEOUT = 10

# Attempts for the on-demand auto lock write, fewer than DEFAULT_ATTEMPTS. The
# write is user-initiated and confirmed by the lock's settings response, so a
# lock that never confirms it should fail fast and report to the user.
AUTO_LOCK_WRITE_ATTEMPTS = 2

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


def retry_bluetooth_connection_error(
    func: WrapFuncType | None = None, *, attempts: int = DEFAULT_ATTEMPTS
) -> Any:
    """
    Define a wrapper to retry on bleak error.

    The accessory is allowed to disconnect us any time so
    we need to retry the operation. Use bare as
    ``@retry_bluetooth_connection_error`` for the default attempt count, or
    ``@retry_bluetooth_connection_error(attempts=N)`` to override it.
    """
    if func is None:
        return functools.partial(retry_bluetooth_connection_error, attempts=attempts)

    async def _async_wrap_retry_bluetooth_connection_error(
        self: PushLock, *args: Any, **kwargs: Any
    ) -> Any:
        _LOGGER.debug("%s: Starting retry loop", self.name)
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
            except RETRYABLE_EXCEPTIONS as err:
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
                # Backoff-class errors (BleakDBusError, DisconnectedError) get
                # a brief pause so the BLE stack can settle before reconnecting.
                backoff = 0.25 if isinstance(err, RETRY_BACKOFF_EXCEPTIONS) else 0
                _LOGGER.debug(
                    "%s: %s error calling %s, retrying in %ss (%s/%s)...",
                    self.name,
                    type(err),
                    func,
                    backoff,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
                if backoff:
                    await asyncio.sleep(backoff)
        return None

    return cast(WrapFuncType, _async_wrap_retry_bluetooth_connection_error)


class _DeadlineWakeup:
    """A wakeup timer serving a monotonic deadline field owned by the caller.

    The deadline field stays the source of truth: every time the timer fires
    the deadline is read again, and a deadline that moved out after the wakeup
    was armed re-arms the timer for the remainder instead of firing early.
    fire runs only once the deadline has truly passed, so the caller never
    has to keep the timer in step with every deadline write.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        get_deadline: Callable[[], float],
        fire: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._get_deadline = get_deadline
        self._fire = fire
        self._timer: asyncio.TimerHandle | None = None

    def arm(self, delay: float | None = None) -> None:
        """Arm the wakeup, at the deadline or after an explicit delay."""
        self.cancel()
        if delay is None:
            delay = max(0.0, self._get_deadline() - time.monotonic())
        self._timer = self._loop.call_later(delay, self._on_timer)

    def cancel(self) -> None:
        """Cancel the wakeup if one is armed."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self) -> None:
        """Fire if the deadline has passed, or re-arm for the remainder."""
        self._timer = None
        remaining = self._get_deadline() - time.monotonic()
        if remaining > 0.0:
            self.arm(remaining)
            return
        self._fire()


class PushLock:
    """A lock with push updates."""

    # Declared here rather than in __init__ because _init_operation_state,
    # which is where they are set, is defined below the methods that write
    # them.
    _pending_op_state: LockStatus | None
    _operation_outcome: LockStatus | None

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
        self.loop = asyncio.get_running_loop()
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
        self._init_operation_state()
        # A booked status poll owes one lock_status() read that the seen set
        # may not suppress: the reading the seen mark records can be the very
        # one the poll exists to replace. Not reset on reconnect, since it is
        # an obligation, not session state.
        self._force_lock_status_poll = False
        self._init_jam_state()
        # Earliest moment an update cycle may read the lock. Every booking is
        # held to it, so a read cannot be taken while the reported state is
        # still settling, whichever path booked the cycle and however short
        # the debounce made it. Not reset on reconnect: a mechanism still
        # moving does not care which link watches it.
        self._earliest_update_time = NEVER_TIME
        self._last_operation_complete_time = NEVER_TIME
        self._always_connected = always_connected
        self._slow_params_set = False
        # Earliest next battery poll attempt (cooldown)
        self._earliest_battery_attempt_time = NEVER_TIME
        # Scheduled battery refresh time (in always_connected mode)
        self._next_battery_refresh_time = NEVER_TIME
        # Auto lock read backoff, mirroring the battery timers above. They
        # persist across reconnects, so a lock that does not answer the read is
        # left alone until its backoff lapses.
        # Earliest next auto lock read after repeated unanswered reads.
        self._earliest_auto_lock_read_time = NEVER_TIME
        # Scheduled auto lock re-read time (in always_connected mode).
        self._next_auto_lock_read_time = NEVER_TIME
        # Consecutive auto lock reads whose READSETTING command timed out.
        self._auto_lock_read_ack_failures = 0
        # Consecutive auto lock reads that were acked but whose 0xBB value
        # never arrived within the response window.
        self._auto_lock_read_response_failures = 0
        # Whether a read has been acked and is still waiting for its 0xBB value,
        # and the deadline by which the value must arrive. Like the counts and
        # timers above, these persist across reconnects so a lock that answers
        # the ack but withholds the value still books its response timeout on the
        # next connection rather than being re-asked forever.
        self._awaiting_auto_lock_response = False
        self._auto_lock_response_deadline = NEVER_TIME

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
            ack_callback=self._ack_callback,
            op_response_callback=self._op_response_callback,
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
            # None of the auto lock read state is reset here: the backoff timers,
            # both failure counts, and the pending-response flag all persist
            # across reconnects, so the hold outlives the connection.
            self._slow_params_set = False
            return self._client

    async def securemode(self) -> None:
        """Set the lock into securemode."""
        await self._run_lock_operation(
            "force_securemode", LockStatus.LOCKING, LockStatus.SECUREMODE
        )

    async def lock(self) -> None:
        """Lock the lock."""
        await self._run_lock_operation(
            "force_lock", LockStatus.LOCKING, LockStatus.LOCKED
        )

    async def unlock(self) -> None:
        """Unlock the lock."""
        await self._run_lock_operation(
            "force_unlock", LockStatus.UNLOCKING, LockStatus.UNLOCKED
        )

    async def unlatch(self) -> None:
        """Unlatch (momentarily open) the lock.

        The op-response arrives when the latch has returned from its open
        dwell, so the completed state is UNLOCKED; a settled UNLATCHED push
        displays only outside an operation window.
        """
        await self._run_lock_operation(
            "force_unlatch", LockStatus.UNLATCHING, LockStatus.UNLOCKED
        )

    def _init_operation_state(self) -> None:
        """Initialise everything that describes the operation in flight.

        Only one is ever in play, the operation lock enforces it, so these
        four describe that one operation: the transitional to stamp when its
        command write reaches the lock, whether any of its attempts got a
        write through, what it learned about the position, and whether its
        display window is open. _run_lock_operation resets the first three at
        the start of each operation and _settle_operation clears the rest at
        the end, so this initialisation is what makes them readable before the
        first one runs.
        """
        self._pending_op_state = None
        self._operation_command_written = False
        self._operation_outcome = None
        self._operation_window_open = False

    async def _run_lock_operation(
        self, op_attr: str, pending_state: LockStatus, complete_state: LockStatus
    ) -> None:
        """Run a lock operation and settle the display once the retries are done.

        This sits outside the retry decorator, so it runs once per operation
        rather than once per attempt, which makes it the only place where the
        operation as a whole has ended. _settle_operation is called from the
        finally, so every way out reaches it, and the attempt exits below
        record what they know in _operation_outcome instead of settling the
        display and booking a status poll each for themselves.

        The unknown position is decided here for the same reason.
        _execute_lock_operation leaves the last attempt's transitional on
        display through a retryable failure, because the next attempt re-stamps
        it at its own write-success; when there is no next attempt that
        transitional would stay on display with no result coming, so the
        position is unknown. The test is that one of our own writes stamped it
        and it is still what the display holds: a write that never got through
        stamped nothing, and between attempts the window is closed, so the lock
        can report a position that replaces the transitional. Both halves are
        needed, because the lock reports the same transitionals for an
        operation someone else started.

        An UnlatchError is the one exit the display cannot answer for. It is
        raised from the write attempt onward, so the latch may have fired
        while the write-success hook never ran and the pre-operation position
        is still on display, reading as a position the operation left alone.
        The exception type is the evidence there.

        A cancelled operation does not take that arm, since CancelledError is a
        BaseException: a cancel is not evidence the lock did or did not move.
        """
        self._cancel_future_update()
        self._operation_outcome = None
        self._operation_command_written = False
        try:
            await self._execute_lock_operation(op_attr, pending_state, complete_state)
        except Exception as err:
            if self._operation_outcome is None and (
                (self._operation_command_written and self.lock_status == pending_state)
                or isinstance(err, UnlatchError)
            ):
                self._operation_outcome = LockStatus.UNKNOWN
            raise
        finally:
            self._settle_operation()

    def _operation_write_success(self) -> None:
        """The command write reached the lock: the single state-action moment.

        Order matters: release any JAMMED display hold first (a fresh operation
        supersedes it), then stamp the operation's transitional while the window
        is still closed, so the stamp passes the filter, then open the window.

        Clearing _seen_jam is the backstop for its invariant, that every
        operation exit applies a jam it recorded, so no record reaches a new
        command's write-success. A record that did reach here would answer the
        same way the hold does: the command a caller issued after the jam is
        the manual intervention a jam calls for, and its outcome is the newer
        truth.
        """
        if time.monotonic() < self._jammed_hold_deadline:
            _LOGGER.debug(
                "%s: New operation write succeeded; releasing the JAMMED display hold",
                self.name,
            )
        self._release_jam_hold()
        self._seen_jam = False
        self._operation_command_written = True
        if self._pending_op_state is not None:
            # Narrowing only: _pending_op_state is typed LockStatus | None and
            # _update_any_state takes an iterable of values, and every path
            # here is inside an operation that set it.
            self._update_any_state([self._pending_op_state], arm_resync=False)
        self._operation_window_open = True

    def _ack_callback(self) -> None:
        """Hold a read off the lock from the acknowledgement of a command.

        Keyed on the opcode, not on whose command it answers. Our own
        operations are the expected source; whether another central's operation
        is ever echoed to us as an acknowledgement is unverified, but if one is,
        the only effect here is a longer hold, which costs a later read.

        The acknowledgement is a moment of its own because the command issue is
        up to ~2 s ahead of it on a slow connection interval, so it is not a
        proxy for the command.
        """
        self._hold_update(LOCK_STALE_STATE_DEBOUNCE_DELAY)

    def _op_response_callback(self) -> None:
        """Hold a read off the lock from an op-response, ours or external.

        An operation started at the lock, in the app, or by auto-lock issues no
        command through us, and whether it is echoed to us as an
        acknowledgement is unverified, so this is the moment to count on for
        one.
        """
        self._hold_update(POST_OP_RESPONSE_DEBOUNCE_DELAY)

    def _close_operation_window(self) -> None:
        """Close the operation window and drop everything it recorded.

        _seen_jam is cleared here rather than at each exit that applies it, so
        the record cannot outlive the window that took it whatever the exit.
        _settle_operation reads it into the outcome before calling this, and
        the retry path reaches this only when it is already clear, so nothing
        that owes the display a jam loses one to the line below.
        """
        self._operation_window_open = False
        self._pending_op_state = None
        self._seen_jam = False

    def _settle_operation(self) -> None:
        """Settle the display and book the status poll: the operation's one exit.

        Every way an operation can end reaches this, a raise and a return
        alike, so the window is closed and, on a lock still being watched, the
        outcome applied and the status poll booked, once and in one place. A
        cancel reaches it too, which is what keeps the window from leaking open
        with no operation in flight: the displayed status would otherwise
        freeze on the transitional, since the filter drops everything the lock
        sends while the window is open.

        _operation_outcome is what the exit knew, and None means it knew
        nothing about where the mechanism is. A jam recorded while the window
        was open outranks it either way: an outcome is a target state inferred
        from the command, while the jam is a reading of where the mechanism
        actually stopped. This is also the only chance to apply that jam,
        because the post-jam register fabricates a plain position, so the
        status poll booked below would put that on display instead, and no later
        signal marks the jam.

        A jam hold still live from an earlier jam outranks the outcome too, by
        the admission rule rather than here: it holds JAMMED on display, so the
        unknown position an unlatch that failed at its write asks for is
        masked. That is the same ruling as the recorded jam above, for the same
        reason, and it is bounded: the hold's wakeup discards the seen mark and
        asks the lock again when the hold lapses.

        Order matters. The window closes first, so the outcome passes the
        filter rather than being dropped by it, and an applied JAMMED
        discharges the record on its way through.

        When the display is left holding a position, the status poll is booked at
        the keep-alive interval, the cadence an always-connected lock polls at
        anyway. When it is left holding anything else, a transitional a cancel
        could not resolve or the unknown of a result that never came, only a
        read can settle it and it is booked at the settle debounce instead.
        That is the earliest such a read is worth taking, because a command
        already at the lock is still driving the motor and a read taken
        immediately returns the pre-operation position. The delay follows from
        the displayed status rather than from the exit taken, so an exit added
        later inherits the answer instead of choosing one.

        What keeps that read out of the settle window is the floor stamped
        here, not the delay asked for below. A booking names the latest moment
        a cycle may run and the deferred-update machinery shortens it freely:
        a pending booking due inside the coalescing interval rewrites this
        request to that interval, and a pending booking due sooner keeps its
        own time. The floor is held by every path that arms a cycle and
        re-applied when one falls due, so the read waits for the mechanism
        however short the booking became.
        """
        outcome = self._operation_outcome
        if self._seen_jam:
            outcome = LockStatus.JAMMED
            _LOGGER.debug(
                "%s: A jam was reported while the operation was in flight; "
                "displaying JAMMED",
                self.name,
            )
        self._close_operation_window()
        if not self._running:
            # The watcher was stopped while this operation was in flight. The
            # window is closed above so the object is left consistent, but
            # neither action below is wanted now: both arm a timer holding the
            # lock past the stop, the booking its own and a jam outcome the
            # display hold's, with nothing left to run either.
            return
        if outcome is not None:
            self._update_any_state([outcome], arm_resync=False)
        self._force_lock_status_poll = True
        self._hold_update(LOCK_STALE_STATE_DEBOUNCE_DELAY)
        self._schedule_future_update_with_debounce(
            KEEP_ALIVE_TIME
            if self.lock_status in POSITION_READINGS
            else LOCK_STALE_STATE_DEBOUNCE_DELAY
        )

    def _init_jam_state(self) -> None:
        """Initialise everything that carries a jam between events.

        _jammed_hold_deadline is the moment a displayed JAMMED stops being
        pinned; every admitted jam event pushes it out again (NEVER_TIME = no
        hold). The deadline and its wakeup survive a reconnect: a jammed
        mechanism outlives the link that reported it, and the wakeup is a
        host-side timer. They do not survive a stop, which releases the hold
        (see _cancel). _seen_jam carries a jam the operation window filtered
        out until the operation applies it, and it is cleared when the window
        closes, so it is discharged well before any reconnect could reach it.
        Outside this initialisation the deadline and the wakeup are written
        only through _arm_jam_hold and _release_jam_hold, so the two cannot
        diverge.
        """
        self._jammed_hold_deadline = NEVER_TIME
        self._jam_hold_wakeup = _DeadlineWakeup(
            self.loop, lambda: self._jammed_hold_deadline, self._jam_hold_ended
        )
        self._seen_jam = False

    def _arm_jam_hold(self, now: float) -> None:
        """Set the hold deadline and arm the wakeup that ends it, as one action.

        Paired with _release_jam_hold: the deadline field and its wakeup are
        written only through this pair, so the two cannot diverge.
        """
        self._jammed_hold_deadline = now + JAMMED_HOLD_TIME
        self._jam_hold_wakeup.arm()

    def _release_jam_hold(self) -> None:
        """Clear the hold deadline and cancel its wakeup, as one action."""
        self._jammed_hold_deadline = NEVER_TIME
        self._jam_hold_wakeup.cancel()

    def _jam_hold_ended(self) -> None:
        """The jam hold has ended: poll the lock's current state.

        The end of a jam cannot be determined remotely, so once the hold has
        lapsed the display's JAMMED is only a guess and one follow-up poll is
        owed. The poll is booked near-term, which is safe where a far-future
        booking was not: _disconnect_with_timer converts a pending deferred
        update into an immediate one at every idle expiry, so a booking
        further out than the idle disconnect is pulled forward repeatedly and
        holds the link up for the whole wait. A near-zero booking runs once,
        right now, on whatever connection state the lock is in.
        """
        if self._operation_lock.locked():
            # No update cycle may be created while an operation runs (it
            # would wait on the operation lock and run the instant the
            # operation ends, inside the settle window). Check again shortly:
            # the retry finds the hold released by the operation's
            # write-success, pushed out by a new jam, or still ended.
            self._jam_hold_wakeup.arm(DEADLINE_WAKEUP_RETRY_DELAY)
            return
        # The jam kept its seen mark because it is a real reading of the
        # position. Discard the mark so the cycle asks the lock instead of
        # trusting the held value.
        self._seen_this_session.discard(LockStatus)
        self._schedule_future_update_with_debounce(0)

    def _admit_lock_status(
        self, incoming: LockStatus, current: LockStatus
    ) -> LockStatus:
        """Decide the displayed lock status for an incoming value.

        Every lock status reaches the state through _update_any_state, whether
        a poll asked for it or the lock pushed it, so this is the one place a
        status is judged; any code path that applies one must pass through here.
        """
        if self._operation_window_open:
            # Literal filter: between our command's write-success and its
            # op-response no received lock status is accepted, whatever the
            # source (our own stale reads, mid-motion readings, foreign
            # centrals, jam evidence). The operation applies its own outcome
            # when it completes. Door and battery members of the same frame
            # are dispatched by type in _update_any_state and are unaffected
            # by this filter.
            if incoming == LockStatus.JAMMED:
                # A dropped jam is not assumed to arrive again. In the jams
                # captured from another central the failure report and the
                # position settle behind it landed within a couple of seconds
                # of each other, close enough for one window to swallow both,
                # and no later frame in those captures carried the jam. The
                # record is what lets the operation apply it at its own exit.
                self._seen_jam = True
            _LOGGER.debug(
                "%s: Operation in flight, not accepting lock status %s",
                self.name,
                incoming,
            )
            return current
        now = time.monotonic()
        if incoming == LockStatus.JAMMED:
            # Admitted jam evidence, whatever the bearer (our own op-response
            # applied after the window closes, a settled 0x07 push, a poll
            # answer, a foreign failure op-response): every event sets a new
            # hold end. A jam is a real reading of the position, so it keeps
            # its seen mark; the wakeup armed here discards the mark and
            # schedules the follow-up poll once the hold actually ends (see
            # _jam_hold_ended). A jam the window filter dropped is discharged
            # here, whichever path carried it back: it has reached the display.
            self._seen_jam = False
            self._arm_jam_hold(now)
            return incoming
        if current == LockStatus.JAMMED and now < self._jammed_hold_deadline:
            # Display hold: after a jam the polled register fabricates a plain
            # locked/unlocked position and no remote signal marks the jam's
            # end, so JAMMED stays pinned until the deadline or until a new
            # operation's write-success releases it. Masking books nothing:
            # the hold's own wakeup carries the follow-up poll, so the hold has
            # no connection side effect.
            _LOGGER.debug(
                "%s: Holding JAMMED, not accepting lock status %s",
                self.name,
                incoming,
            )
            return current
        return incoming

    # The two wrappers run in the reverse of the order they read: operation_lock
    # holds self._operation_lock for the whole call, and inside it
    # retry_bluetooth_connection_error runs this body up to DEFAULT_ATTEMPTS
    # times. Another attempt follows only from an AuthError or a member of
    # RETRYABLE_EXCEPTIONS; every other type, the operation errors below
    # included, leaves on its first raise.
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
        # Re-set on every retry attempt: the transitional to stamp when this
        # attempt's command write reaches the lock (write-success).
        self._pending_op_state = pending_state
        try:
            lock = await self._ensure_connected()
            self._cancel_future_update()
            # Re-stamped per attempt, so a retry moves the floor forward with
            # it rather than leaving the first attempt's window to expire.
            self._hold_update(LOCK_STALE_STATE_DEBOUNCE_DELAY)
            # Hand the write-success hook to this operation alone. The window it
            # opens is closed only on the paths below, so nothing that did not
            # come through here can open it.
            await getattr(lock, op_attr)(self._operation_write_success)
        except OperationFailedError:
            # The op-response arrived and its result byte named a failure
            # (mechanical codes are a motor stall, the rest name their own
            # cause): the exchange completed, the operation did not. The
            # parser's JAMMED emission fell inside our own window, so the
            # operation records it as the outcome and the settle applies it,
            # then propagates: the display carries the jam, the exception tells
            # the caller the operation did not happen.
            self._operation_outcome = LockStatus.JAMMED
            _LOGGER.debug(
                "%s: %s reported failure; displaying JAMMED", self.name, op_attr
            )
            self._complete_operation(time.monotonic())
            raise
        except (OperationIncompleteError, UnlatchError):
            # Non-retryable: this propagates to the caller. No outcome is
            # recorded because this exit has no evidence of the position: our
            # write may have succeeded, leaving a transitional on display with
            # no result coming, which _run_lock_operation answers with the
            # unknown position. An UnlatchError is raised from the write
            # attempt onward, so the latch may have fired even where the
            # write-success hook never ran; _run_lock_operation answers that
            # one the same way, from the exception type rather than from the
            # display.
            _LOGGER.debug(
                "%s: %s did not complete; the result never arrived",
                self.name,
                op_attr,
            )
            raise
        except Exception as ex:
            if self._seen_jam:
                # The attempt ladder ends here. These types mean the command
                # may not have been delivered, so the next attempt would send
                # it again, driving the motor into a mechanism the lock has
                # just reported jammed; clearing a jam is work for a person at
                # the lock. OperationIncompleteError carries that to the
                # caller: it is outside the retry set, and it states what is
                # true of this operation, that its result never arrived.
                raise OperationIncompleteError(
                    f"{self.name}: a jam was reported while {op_attr} was in "
                    f"flight; the command was not re-sent and the result is "
                    f"unknown"
                ) from ex
            # Retryable (or terminal after the retries run out): close the
            # window so normal acceptance resumes; the next attempt re-stamps at
            # its own write-success. No UNKNOWN stamp here: through send and
            # acknowledgement stage retries the display stays untouched, and
            # _run_lock_operation settles it if the attempts run out.
            self._close_operation_window()
            _LOGGER.debug(
                "%s: Failed to execute lock operation due to %s",
                self.name,
                ex,
            )
            raise
        self._operation_outcome = complete_state
        _LOGGER.debug("%s: Finished %s", self.name, complete_state)
        self._complete_operation(time.monotonic())

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
            await self._set_auto_lock_or_warn(AutoLockMode.OFF, 0)
            return

        duration = AUTO_LOCK_DEFAULT_DURATION
        if self.auto_lock and self.auto_lock.mode != AutoLockMode.OFF:
            duration = self.auto_lock.duration
        elif self.auto_lock_prev and self.auto_lock_prev.mode != AutoLockMode.OFF:
            # If the auto lock is currently off, use the previous duration
            duration = self.auto_lock_prev.duration
        await self._set_auto_lock_or_warn(mode, duration)

    async def set_auto_lock_duration(self, duration: int) -> None:
        """Set auto lock setting."""
        if duration == 0:
            if self.auto_lock and self.auto_lock.mode == AutoLockMode.OFF:
                _LOGGER.debug("%s: Auto lock is already off", self.name)
                return
            await self._set_auto_lock_or_warn(AutoLockMode.OFF, 0)
            return

        mode = AutoLockMode.TIMER
        if self.auto_lock and self.auto_lock.mode != AutoLockMode.OFF:
            mode = self.auto_lock.mode
        elif self.auto_lock_prev and self.auto_lock_prev.mode != AutoLockMode.OFF:
            # If the auto lock is currently off, use the previous mode
            mode = self.auto_lock_prev.mode
        await self._set_auto_lock_or_warn(mode, duration)

    async def _set_auto_lock_or_warn(self, mode: AutoLockMode, duration: int) -> None:
        """Set auto lock, surfacing a write the lock never confirmed."""
        try:
            await self._set_auto_lock(mode, duration)
        except TimeoutError as err:
            _LOGGER.warning(
                "%s: Lock did not confirm the auto lock setting write "
                "after %s attempts; the lock may not support auto lock",
                self.name,
                AUTO_LOCK_WRITE_ATTEMPTS,
            )
            raise TimeoutError(
                f"{self.name}: Lock did not confirm the auto lock setting write"
            ) from err

    @retry_bluetooth_connection_error(attempts=AUTO_LOCK_WRITE_ATTEMPTS)
    async def _set_auto_lock(self, mode: AutoLockMode, duration: int) -> None:
        """Set auto lock setting."""
        if not self._running:
            raise RuntimeError(
                f"{self.name}: Set auto lock operation not possible because not running"
            )
        # Duration validation
        if duration not in self.auto_lock_durations:
            raise ValueError(f"Invalid auto lock duration: {duration}")
        # Unlike lock/unlock/securemode, this path does not optimistically mutate
        # _lock_state.auto_lock, so there is no prior value to restore on failure.
        # Notify callbacks or the next poll surface the authoritative state.
        try:
            lock = await self._ensure_connected()
            self._cancel_future_update()
            await lock.set_auto_lock(mode, duration)
            # A confirmed write both proves the lock supports auto lock (clear
            # any failure backoff) and changes the value: drop AutoLockState and
            # both deadlines so the next update reads the new value straight
            # back, confirming the write and refreshing the display.
            self._auto_lock_read_ack_failures = 0
            self._auto_lock_read_response_failures = 0
            self._awaiting_auto_lock_response = False
            self._earliest_auto_lock_read_time = NEVER_TIME
            self._next_auto_lock_read_time = NEVER_TIME
            self._seen_this_session.discard(AutoLockState)
        except Exception as ex:
            # The retry_bluetooth_connection_error wrapper calls
            # _async_handle_disconnected for RETRY_EXCEPTIONS /
            # RETRY_BACKOFF_EXCEPTIONS only; AuthError, BleakNotFoundError and
            # any other exception propagate without disconnecting.
            _LOGGER.debug(
                "%s: Failed to execute set auto lock operation due to %s",
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

    def _update_any_state(
        self,
        states: Iterable[LockStateValue | AuthState],
        arm_resync: bool = True,
    ) -> None:
        """Apply states to the display.

        arm_resync is False for the states an operation applies itself. A
        status change coming from the lock arms a resync cycle to read the
        settled value back; a status the operation stamped needs no such read,
        and arming one from inside an operation creates a cycle that waits on
        the operation lock and runs the instant the operation ends, inside the
        settle window. Those states are read by the operation's own follow-up
        status poll instead (see _settle_operation).
        """
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
                # Route every incoming lock status through the policy before the
                # equality check: a repeated identical reading must still reach
                # the admission filter, because a repeated JAMMED re-arms the
                # display hold deadline (see _admit_lock_status), so short-
                # circuiting an unchanged value would let the hold lapse.
                admitted = self._admit_lock_status(state, lock_state.lock)
                if admitted is not state or admitted not in POSITION_READINGS:
                    # The seen set suppresses the follow-up lock_status() poll
                    # in _update, so it may only record a reading we hold. A
                    # reading the policy discarded is not a reading we hold, and
                    # a value the lock is not holding still says nothing about
                    # where the mechanism will stop, so in both cases the poll
                    # has to stay armed.
                    #
                    # _admit_lock_status returns current, which is the same
                    # singleton as the incoming value when the two are equal, so
                    # an equal-but-filtered reading counts as admitted here.
                    # That is deliberate and harmless: the displayed value
                    # already is the reading.
                    self._seen_this_session.discard(state_type)
                if lock_state.lock != admitted:
                    changes["lock"] = admitted
            elif isinstance(state, DoorStatus):
                if lock_state.door != state:
                    changes["door"] = state
            elif isinstance(state, BatteryState):
                if state.voltage <= 3.0:
                    # A refused reading is not a reading, so it must not count
                    # as having seen the battery this session: the seen mark
                    # suppresses the next poll, and the reading it would
                    # suppress it for was never published.
                    self._seen_this_session.discard(BatteryState)
                    # A checksum-clean frame reporting 3.0 V or less is
                    # 0.75 V per cell, against a table that already treats
                    # 1.24 V per cell as empty. That is unexpected lock
                    # behavior, so it is surfaced rather than silently
                    # dropped.
                    _LOGGER.warning(
                        "%s: Battery voltage is impossible: %s",
                        self.name,
                        state.voltage,
                    )
                    continue
                if lock_state.battery != state:
                    changes["battery"] = state
            elif isinstance(state, AutoLockState):
                # The 0xBB settings response arriving here carries the stored
                # value and is the success signal for the read backoff: clear
                # both failure counts, disarm the pending-response deadline the
                # ack armed, and arm the refresh timer where the value lands.
                self._auto_lock_read_ack_failures = 0
                self._auto_lock_read_response_failures = 0
                self._awaiting_auto_lock_response = False
                self._earliest_auto_lock_read_time = NEVER_TIME
                self._next_auto_lock_read_time = (
                    time.monotonic() + AUTO_LOCK_READ_REFRESH_INTERVAL
                )
                if lock_state.auto_lock != state:
                    changes["auto_lock"] = state
                    changes["auto_lock_prev"] = lock_state.auto_lock
            else:
                raise ValueError(f"Unexpected state type: {state}")

        if not changes:
            return

        lock_state = replace(lock_state, **changes)
        if (
            arm_resync
            and original_lock_status != lock_state.lock
            and (not lock_state.auth or lock_state.auth.successful)
            and original_lock_status != LockStatus.UNKNOWN
        ):
            self._schedule_future_update(RESYNC_DELAY)

        self._callback_state(lock_state)

    def _record_auth_success(self) -> None:
        """Record a successful round trip with the lock.

        Resets the consecutive-failure count that arms the auth latch, and
        publishes the auth state itself. This is the only producer of
        AuthState(successful=True); the latch in the retry decorator is the
        only producer of the failure, so both publish through the same path.
        _update_any_state drops a repeat, so a keep-alive cycle that changes
        nothing publishes nothing.
        """
        _AUTH_FAILURE_HISTORY.auth_success(self.address)
        self._update_any_state([AuthState(successful=True)])

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

    async def _poll_battery(self, lock: Lock) -> bool:
        """Poll battery if needed: periodic refresh, timeout cooldown, errors.

        Battery state requires a poll of the lock to update. In always_connected mode
        _seen_this_session never clears, so once the refresh deadline passes
        BatteryState is evicted to force a re-poll -- but only after the cooldown gate.

        The read is issued for its effect on the receive path, which applies
        and publishes the value before the await here resolves, so the
        fetched value is discarded.

        Returns whether a request was made.
        """
        assert self._lock_info is not None  # nosec
        if self._lock_info.model in NO_BATTERY_SUPPORT_MODELS:
            _LOGGER.debug(
                "%s: Needs battery workaround model %s",
                self.name,
                self._lock_info.model,
            )
            return False

        now = time.monotonic()
        # Skip while in cooldown after a prior battery timeout.
        if now < self._earliest_battery_attempt_time:
            _LOGGER.debug(
                "%s: Skipping battery request due to recent timeout "
                "(cooldown until %.1fs)",
                self.name,
                self._earliest_battery_attempt_time - now,
            )
            return False

        # Periodic refresh: evict BatteryState once its deadline has passed.
        if (
            self._always_connected
            and BatteryState in self._seen_this_session
            and now > self._next_battery_refresh_time
        ):
            self._seen_this_session.discard(BatteryState)
        if BatteryState in self._seen_this_session:
            return False

        try:
            await lock.battery()
            self._record_auth_success()
            # Success: disable cooldown and schedule the next refresh.
            self._earliest_battery_attempt_time = NEVER_TIME
            self._next_battery_refresh_time = now + BATTERY_REFRESH_INTERVAL
        except TimeoutError as err:
            _LOGGER.info(
                "%s: Battery request timed out (%s), will retry in %d "
                "seconds. Continuing with other updates.",
                self.name,
                err,
                BATTERY_TIMEOUT_COOLDOWN,
            )
            # Set cooldown to prevent repeated timeouts.
            self._earliest_battery_attempt_time = now + BATTERY_TIMEOUT_COOLDOWN
        except BleakError as err:
            _LOGGER.debug(
                "%s: Battery request failed (%s), continuing with other updates.",
                self.name,
                err,
            )

        return True

    async def _probe_lock_info(self, lock: Lock) -> LockInfo:
        """Probe the lock for info, falling back to defaults on failure."""
        try:
            lock_info = await lock.lock_info()
        except (TimeoutError, BleakError) as err:
            _LOGGER.warning(
                "%s: Failed to probe lock info (%s), continuing with defaults",
                self.name,
                err,
            )
            lock_info = LockInfo(
                manufacturer="Unknown",
                model="",
                serial=self.address,
                firmware="Unknown",
            )
        _LOGGER.debug("Obtained lock info: %s", lock_info)
        return lock_info

    def _arm_auto_lock_read_backoff_if_exhausted(self, now: float) -> bool:
        """Arm the read backoff once the failure count hits the limit.

        A lock shows one shape at a time -- either silent to the command (ack
        timeouts) or acking without the value (response timeouts) -- so only one
        counter grows. Summing them lets whichever it is trip the one threshold
        after AUTO_LOCK_READ_FAILURE_THRESHOLD failures in a row.
        """
        total = (
            self._auto_lock_read_ack_failures + self._auto_lock_read_response_failures
        )
        if total < AUTO_LOCK_READ_FAILURE_THRESHOLD:
            return False
        self._earliest_auto_lock_read_time = now + AUTO_LOCK_READ_FAILURE_BACKOFF
        _LOGGER.info(
            "%s: Auto lock setting request unresolved after %s attempts "
            "(%s ack timeouts, %s response timeouts); the lock may not support "
            "auto lock; not asking again for %d seconds",
            self.name,
            total,
            self._auto_lock_read_ack_failures,
            self._auto_lock_read_response_failures,
            AUTO_LOCK_READ_FAILURE_BACKOFF,
        )
        self._auto_lock_read_ack_failures = 0
        self._auto_lock_read_response_failures = 0
        return True

    async def _read_auto_lock_setting(self, lock: Lock) -> bool:
        """Request the auto lock setting; return whether a read was issued.

        Mirrors _poll_battery's two-timer backoff. The solicited wait returns
        the READSETTING acknowledgment, whose value field is a fixed zero; the
        stored setting arrives afterwards on the notify path as the 0xBB
        settings response. Issue the read only to trigger that response, which
        the notify path decodes and applies; its arrival is the success signal
        that arms the refresh timer (see _update_any_state).

        A lock that never gives the value back is caught two ways, both counting
        toward AUTO_LOCK_READ_FAILURE_THRESHOLD. A lock silent to the command
        times out on the ack; a lock that acks but withholds the 0xBB leaves the
        response deadline armed, and the next cycle sees it lapse with the value
        still unseen. Either way, after THRESHOLD failures in a row the read
        backs off for AUTO_LOCK_READ_FAILURE_BACKOFF seconds. All of this state
        lives outside the reconnect reset so it survives disconnects.
        """
        now = time.monotonic()
        # Skip while backed off after repeated unanswered reads.
        if now < self._earliest_auto_lock_read_time:
            return False
        # Resolve a read that was acked last cycle but is still waiting for its
        # 0xBB value. The 0xBB clears this flag on the notify path the moment it
        # lands, so if it is still set the value has not arrived.
        if self._awaiting_auto_lock_response:
            if now <= self._auto_lock_response_deadline:
                # The value may still be in flight; do not re-read or strike.
                return False
            # Acked but the value never came: a response timeout, counted like
            # an ack timeout. Fall through to re-read unless it armed the backoff.
            self._awaiting_auto_lock_response = False
            self._auto_lock_read_response_failures += 1
            if self._arm_auto_lock_read_backoff_if_exhausted(now):
                return False
        # Periodic refresh: evict AutoLockState once its deadline has passed so
        # the next cycle re-reads (always_connected only, as battery does).
        if (
            self._always_connected
            and AutoLockState in self._seen_this_session
            and now > self._next_auto_lock_read_time
        ):
            self._seen_this_session.discard(AutoLockState)
        if AutoLockState in self._seen_this_session:
            return False
        try:
            await lock.auto_lock_status()
        except TimeoutError:
            # Handle the timeout here, as _poll_battery does. A timeout that
            # reaches _update's retry decorator is read as a lost connection and
            # forces a reconnect; catching it locally keeps the connection up
            # and the read on its backoff.
            self._auto_lock_read_ack_failures += 1
            self._arm_auto_lock_read_backoff_if_exhausted(now)
            return False
        except BleakError as err:
            # Mirror _poll_battery: a transport fault leaves the backoff alone
            # (only a timeout arms it) and the update continues. A persistent
            # fault surfaces on a later status poll or the disconnect callback.
            _LOGGER.debug(
                "%s: Auto lock setting request failed (%s), "
                "continuing with other updates.",
                self.name,
                err,
            )
            return False
        # The ack arrived; expect the 0xBB value within the response window --
        # unless it already landed on the notify path during the await (same
        # loop turn), in which case AutoLockState is already seen and there is
        # nothing to wait for, so do not arm a deadline for a value in hand.
        if AutoLockState not in self._seen_this_session:
            self._awaiting_auto_lock_response = True
            self._auto_lock_response_deadline = now + AUTO_LOCK_READ_RESPONSE_TIMEOUT
        return True

    @operation_lock
    @retry_bluetooth_connection_error
    async def _update(self) -> LockState:
        """Update the lock state.

        Each read publishes through the receive path as its answer lands, so
        the return value is the live state after the cycle rather than a copy
        this method assembled. Both callers discard it.
        """
        has_lock_info = self._lock_info is not None

        _LOGGER.debug(
            "%s: Starting update (has_lock_info: %s)", self.name, has_lock_info
        )
        lock = await self._ensure_connected()
        if not self._lock_info:
            self._lock_info = await self._probe_lock_info(lock)

        # Each read below is issued for its effect on the receive path:
        # session._notify hands every frame to the state path before it
        # resolves the waiter the read is blocked on, so the answer has
        # already been applied and published by the time the await returns.
        # The fetched values therefore carry nothing new and are discarded.
        made_request = False

        # Asking for battery first seems to reduce the chance of the lock
        # getting into a bad state.
        if await self._poll_battery(lock):
            made_request = True

        if (
            DoorStatus not in self._seen_this_session
            and self._lock_info
            and self._lock_info.door_sense
        ):
            made_request = True
            await lock.door_status()
            self._record_auth_success()

        if await self._read_auto_lock_setting(lock):
            made_request = True
            self._record_auth_success()

        # Only ask for the lock status if we haven't seen
        # it this session since notify callbacks will happen
        # if it changes and the extra polling can cause the lock
        # to get into a bad state.
        #
        # However, we always want to poll lock
        # state to keep the connection alive if we are always connected.
        #
        # A status poll booked after an operation asks regardless: the seen
        # mark may record the very reading the poll exists to replace.
        if (
            self._force_lock_status_poll
            or LockStatus not in self._seen_this_session
            or (not made_request and self._always_connected)
        ):
            made_request = True
            await lock.lock_status()
            # One-shot, discharged only by a read that answered: a cycle that
            # failed before this point leaves the obligation for the retry.
            self._force_lock_status_poll = False
            self._record_auth_success()

        _LOGGER.debug("%s: Finished update", self.name)

        current = self._get_current_state()
        # One publish per cycle, of the live state. Every reading this cycle
        # took was applied and published as its frame landed, so this hands
        # back exactly what is already held and cannot change any member.
        #
        # It exists because a consumer may read the callback as a liveness
        # signal rather than only as a change notification. The Home Assistant
        # entities do: they mark themselves unavailable from their own
        # advertisement tracking and have no path back other than this
        # callback, so a cycle that read the same values as the last one still
        # has to report.
        self._callback_state(current)

        if not has_lock_info:
            # On first update free up the connection
            # so we can bring other locks online if
            # the bluetooth adapter is out of connections
            # slots. We reset the timer to a low number
            # so that if another update request is pending
            # we do not disconnect until it completes.
            self._next_disconnect_delay = FIRST_CONNECTION_DISCONNECT_TIME
            self._reset_disconnect_timer()

        if self._always_connected and made_request:
            await self._set_slow_connection_params(lock)

        if made_request:
            self._last_operation_complete_time = time.monotonic()
            self._reschedule_next_keep_alive()
        return self._get_current_state()

    async def _set_slow_connection_params(self, lock: Lock) -> None:
        """Set slow BLE connection parameters to conserve battery."""
        if self._slow_params_set:
            return
        client = lock.client
        if client is None:
            return
        try:
            await client.set_connection_params(
                SLOW_MIN_INTERVAL, SLOW_MAX_INTERVAL, SLOW_LATENCY, SLOW_TIMEOUT
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.debug(
                "%s: Failed to set connection parameters", self.name, exc_info=True
            )
        else:
            self._slow_params_set = True
            _LOGGER.debug("%s: Set slow connection parameters", self.name)

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
        # An empty payload is skipped rather than indexed: the advertisement is
        # radio input and its length is not ours to assume.
        if apple_data := mfr_data.get(APPLE_MFR_ID):
            first_byte = apple_data[0]
            if first_byte == HAP_FIRST_BYTE and len(apple_data) >= HAP_STATE_RECORD_END:
                hk_state = get_homekit_state_num(apple_data)
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
        # Yale YALE_MFR_ID advertisements come in two formats:
        # - 1-byte: lock state toggle (0/1), used for change detection
        # - 18-byte: 2 header bytes + the lock's 16-byte cloud ID (the
        #   identifier used by the Yale/ASSA ABLOY cloud API), e.g.
        #   b'\x00\x00\x80\x15\xd0\x11\xf7\xa5\x43\x1f\x85\xd7\xff\x23\x5f\x1e\x75\x46'
        # Only track byte[0] from the 1-byte format. The two formats
        # alternate every few seconds; without the length check, the
        # static 0x00 header of the 18-byte format causes repeated
        # connections if it differs from the 1-byte value.
        is_first_advertisement = self._last_adv_value == -1
        # As above, an empty payload is skipped rather than indexed.
        if (yale_data := mfr_data.get(YALE_MFR_ID)) and (
            len(yale_data) == 1 or is_first_advertisement
        ):
            current_value = yale_data[0]
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
        # The hold exists to make a user aware of a jam through the UI, and a
        # stopped watcher has no display to hold it on. Releasing it takes the
        # wakeup with it, which would otherwise fire after the stop and book a
        # cycle on a lock nothing is watching. Leaving the deadline behind
        # instead would mask the lock's status on a restarted watcher with no
        # wakeup left to end the hold or to ask the lock again.
        self._release_jam_hold()
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

    def _hold_update(self, seconds: float) -> None:
        """Hold any update cycle off the lock for seconds from now.

        Forward only, because the moments that set the floor interleave: an
        op-response asking for its shorter hold must not pull a read forward
        into a window the acknowledgement before it already claimed.
        """
        self._earliest_update_time = max(
            self._earliest_update_time, time.monotonic() + seconds
        )

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
        """Schedule an update in future seconds, never before the floor.

        Held here rather than at the callers because every arming path passes
        through, including the ones that shorten a request. The floor is kept
        as a moment and the booking as a delay, so it is converted here and
        never carried across the two.
        """
        future_update_time = max(
            future_update_time, self._earliest_update_time - time.monotonic()
        )
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
        if now < self._earliest_update_time:
            # A booking armed before the floor moved, so it is re-armed for the
            # remainder. Unconditionally: the timer that brought us here has
            # already been cancelled, so there is nothing to coalesce with, and
            # this booking is the one the floor is for.
            _LOGGER.debug("%s: Rescheduling update to avoid stale state", self.name)
            self._schedule_future_update(self._earliest_update_time - now)
            return
        if self._operation_lock.locked():
            # The floor is read now, but a cycle created here would not reach
            # the lock until the operation released it, and the settle moves the
            # floor forward as the operation ends. The cycle would then run the
            # instant the operation ends, inside the settle window it just
            # opened, because a cycle already created never reads the floor
            # again. Check again shortly instead: the retry finds the operation
            # finished, the floor moved out, or the moment still passed.
            _LOGGER.debug(
                "%s: Rescheduling update until the operation lock is released",
                self.name,
            )
            self._schedule_future_update_with_debounce(DEADLINE_WAKEUP_RETRY_DELAY)
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
