from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import struct
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, replace
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
    MANUAL_INTERVENTION_STATUSES,
    SETUP_CONDITION_STATUSES,
    YALE_MFR_IDS,
    ActivityValue,
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
from .lock import ActivityLogOverrunError, Lock, validate_keycode_slot
from .session import (
    AuthError,
    BluetoothError,
    DisconnectedError,
    NoAdvertisementError,
    OperationFailedError,
    OperationIncompleteError,
    ResponseError,
    YaleXSBLEError,
)
from .util import (
    asyncio_timeout,
    encode_keycode_pin,
    is_disconnected_error,
    local_name_is_unique,
)

_LOGGER = logging.getLogger(__name__)

# Advertisement debugger (this one is quite noisy so it has its only logger)
_ADV_LOGGER = logging.getLogger("yalexs_ble_adv")

WrapFuncType = TypeVar("WrapFuncType", bound=Callable[..., Any])
_T = TypeVar("_T")

# A monotonic timestamp ~one day in the past, used as a "never happened /
# no deadline" sentinel. Makes no assumption about the clock's epoch.
NEVER_TIME = time.monotonic() - 86400.0

DEFAULT_ATTEMPTS = 4

# How long to wait to disconnect after an operation
DISCONNECT_DELAY = 5.1

# How long to wait to disconnect after an operation if there is a pending update
DISCONNECT_DELAY_PENDING_UPDATE = 12.5

# The lock drops an idle link after this long.
IDLE_DISCONNECT_TIME = 30.0
KEEP_ALIVE_TIME = 25.0

# Reconnect delay while updates keep failing; doubles per failure up to the
# cap. A state-changing advertisement still pulls an update earlier.
RECONNECT_BACKOFF_TIME = 2.0
MAX_RECONNECT_BACKOFF_TIME = 60.0

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

# How long after the lock last reported a jam or setup condition the report
# takes precedence over a command of ours.
JAMMED_PRECEDENCE_TIME = 5.0

# How long a reported jam or setup condition stays on display; a poll soon
# after a jam may report a position the mechanism is not in.
JAMMED_HOLD_TIME = 30.0

# How long to wait before re-checking while an operation holds the lock.
OPERATION_IN_PROGRESS_DEFER_SECONDS = 1.0

# How long to wait if we get an update storm from the lock
UPDATE_IN_PROGRESS_DEFER_SECONDS = DISCONNECT_DELAY - 1

# Statuses that report a position the lock is holding; the attention statuses
# qualify because they end only by hand, and UNLATCHED because the lock holds
# it for the dwell. Any other status must stay out of _seen_this_session so the
# follow-up lock_status() poll runs.
POSITION_READINGS = MANUAL_INTERVENTION_STATUSES | {
    LockStatus.LOCKED,
    LockStatus.UNLOCKED,
    LockStatus.UNLATCHED,
    LockStatus.SECUREMODE,
}

# Statuses the lock reports while the motor runs.
TRANSITIONAL_READINGS = frozenset(
    {LockStatus.LOCKING, LockStatus.UNLOCKING, LockStatus.UNLATCHING}
)

RETRY_BACKOFF_EXCEPTIONS = (BleakDBusError, DisconnectedError)

RETRY_EXCEPTIONS = (ResponseError, *BLEAK_RETRY_EXCEPTIONS)

RETRYABLE_EXCEPTIONS = (*RETRY_BACKOFF_EXCEPTIONS, *RETRY_EXCEPTIONS)

# 255 seems to be broadcast randomly when
# there is no update from the lock.
VALID_ADV_VALUES = {0, 1}

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

# Consecutive failed activity drains before the drain is left alone for a
# while; a lock whose log never ends would otherwise cost 32 reads per update.
ACTIVITY_DRAIN_FAILURE_THRESHOLD = 3
ACTIVITY_DRAIN_FAILURE_BACKOFF = 3600
# Delay before the first drain, so the first update is not held up by the
# backlog and has freed the adapter slot before the lock is asked again.
ACTIVITY_PRIME_DELAY = 10.0
# Consecutive drains that read records but never reach the end marker before
# the log counts as one that never ends rather than a deep backlog.
ACTIVITY_DRAIN_MAX_OVERRUNS = 8

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

AUTO_LOCK_DEFAULT_DURATION = 90


def _select_yale_payload(
    mfr_data: Mapping[int, bytes],
) -> tuple[int | None, bytes | None]:
    """Choose the payload to read, and its company ID, from the Yale IDs present.

    A 1-byte payload first, then any non-empty one, 465 first on a tie. Both
    None when no Yale ID is present.
    """
    present = [
        (mfr_id, mfr_data[mfr_id]) for mfr_id in YALE_MFR_IDS if mfr_id in mfr_data
    ]
    if not present:
        return None, None
    # The scanner merges manufacturer data per company ID, so the payload under
    # one ID may be a cached frame while the current value arrives under the other.
    return min(present, key=lambda item: (len(item[1]) != 1, not item[1]))


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
    func: WrapFuncType | None = None,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    lock_operation: bool = False,
) -> Any:
    """
    Define a wrapper to retry on bleak error.

    The accessory is allowed to disconnect us any time so
    we need to retry the operation. Use bare as
    ``@retry_bluetooth_connection_error`` for the default attempt count, or
    ``@retry_bluetooth_connection_error(attempts=N)`` to override it.
    ``lock_operation=True`` ends the attempts as ``OperationIncompleteError``
    instead of retrying while a reported jam or setup condition takes precedence.
    """
    if func is None:
        return functools.partial(
            retry_bluetooth_connection_error,
            attempts=attempts,
            lock_operation=lock_operation,
        )

    async def _async_wrap_retry_bluetooth_connection_error(
        self: PushLock, *args: Any, **kwargs: Any
    ) -> Any:
        _LOGGER.debug("%s: Starting retry loop", self.name)
        max_attempts = attempts - 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return await func(self, *args, **kwargs)
            except AuthError as err:
                last_error = err
                _AUTH_FAILURE_HISTORY.auth_failed(self.address)
                if _AUTH_FAILURE_HISTORY.should_raise(self.address):
                    # If the bluetooth connection drops in the middle of authentication
                    # we may see it as a failed authentication. If we see 5 failed
                    # authentications in a row we can reasonably assume that the key has
                    # changed and we should re-authenticate.
                    self._update_any_state([AuthState(successful=False)])
                    raise
                backoff = 0.25
            except BleakNotFoundError:
                # The lock cannot be found so there is no
                # point in retrying.
                raise
            except RETRYABLE_EXCEPTIONS as err:
                last_error = err
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
            if attempt < max_attempts:
                # Never re-send into a mechanism the lock reported as needing
                # attention, whatever position it has reported since.
                if lock_operation:
                    self._end_a_retry_under_precedence(last_error)
                _LOGGER.debug(
                    "%s: %s error calling %s, retrying in %ss (%s/%s)...",
                    self.name,
                    type(last_error),
                    func,
                    backoff,
                    attempt,
                    max_attempts,
                    exc_info=last_error,
                )
                if backoff:
                    await asyncio.sleep(backoff)
        # Below the re-auth latch an auth failure is read as a link dropped
        # mid-authentication, so the exhausted attempts fail as a disconnect.
        raise DisconnectedError(str(last_error)) from last_error

    return cast(WrapFuncType, _async_wrap_retry_bluetooth_connection_error)


# The secure lock a reported status settles to; a status absent here applies
# to both locks.
_SECURE_FOR_STATUS = {
    LockStatus.SECUREMODE: LockStatus.LOCKED,
    LockStatus.LOCKING: LockStatus.UNLOCKED,
    LockStatus.LOCKED: LockStatus.UNLOCKED,
    LockStatus.UNLOCKED: LockStatus.UNLOCKED,
    LockStatus.UNLATCHED: LockStatus.UNLOCKED,
}


@dataclass(frozen=True)
class _Display:
    """The statuses shown on the main and secure locks."""

    lock: LockStatus
    secure: LockStatus

    @property
    def settled(self) -> bool:
        """Whether both locks show a position the lock holds."""
        return {self.lock, self.secure} <= POSITION_READINGS


def _project_lock_status(reported: LockStatus, state: LockState) -> _Display:
    """Project a reported status onto the display.

    The main lock keeps SECUREMODE for compatibility.
    """
    if reported in (LockStatus.UNLOCKING, LockStatus.UNLATCHING):
        # A secured lock's secure channel moves too; a retry's second stamp
        # keeps it moving.
        moving = state.secure in (LockStatus.LOCKED, LockStatus.UNLOCKING)
        secure = LockStatus.UNLOCKING if moving else LockStatus.UNLOCKED
        return _Display(reported, secure)
    return _Display(reported, _SECURE_FOR_STATUS.get(reported, reported))


def _securing_display(state: LockState) -> _Display:
    """Securing a lock that is already locked moves only the secure lock."""
    if state.lock in (LockStatus.LOCKED, LockStatus.SECUREMODE):
        return _Display(state.lock, LockStatus.LOCKING)
    return _Display(LockStatus.LOCKING, LockStatus.LOCKING)


@dataclass(frozen=True)
class _LockOperation:
    """A mechanical command with its transitional display and settled status."""

    op_attr: str
    pending: Callable[[LockState], _Display]
    complete: LockStatus


_LOCK_OPERATION = _LockOperation(
    "force_lock",
    functools.partial(_project_lock_status, LockStatus.LOCKING),
    LockStatus.LOCKED,
)
_UNLOCK_OPERATION = _LockOperation(
    "force_unlock",
    functools.partial(_project_lock_status, LockStatus.UNLOCKING),
    LockStatus.UNLOCKED,
)
_SECUREMODE_OPERATION = _LockOperation(
    "force_securemode", _securing_display, LockStatus.SECUREMODE
)
_UNLATCH_OPERATION = _LockOperation(
    "force_unlatch",
    functools.partial(_project_lock_status, LockStatus.UNLATCHING),
    LockStatus.UNLATCHED,
)


class PushLock:
    """A lock with push updates."""

    # Declared so mypy does not infer None from _init_operation_state.
    _operation_outcome: LockStatus | None
    _operation_failure: Exception | None

    def __init__(  # noqa: PLR0915
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
        self._activity_callbacks: list[Callable[[ActivityValue], None]] = []
        self._activity_primed = False
        self._activity_drain_pending = False
        self._activity_drain_failures = self._activity_overruns = 0
        self._earliest_activity_drain_time = NEVER_TIME
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
        # The next cycle reads lock_status() even if _seen_this_session would
        # skip it, since that reading may be the one to replace.
        self._force_lock_status_poll = False
        self._init_jam_state()
        # Every scheduled cycle is held to this, so none reads the lock while
        # the reported state is still settling.
        self._earliest_update_time = NEVER_TIME
        self._last_operation_complete_time = NEVER_TIME
        self._reconnect_backoff = 0.0
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
    def secure_status(self) -> LockStatus:
        """Return the current status of the secure lock."""
        return self._lock_state.secure if self._lock_state else LockStatus.UNKNOWN

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

    def register_activity_callback(
        self, callback: Callable[[ActivityValue], None]
    ) -> Callable[[], None]:
        """Register a callback for each new lock activity record.

        Records are read on the next update after a lock or door status change,
        or a change of the advertised activity flag.
        """
        if not self._activity_callbacks:
            # The first registration primes, so the backlog is not delivered as new.
            self._activity_primed = False
            self._activity_drain_pending = True
            if self._running and self._lock_info is not None:
                # Past the first update, so nothing else would prime soon.
                self._schedule_future_update_with_debounce(ACTIVITY_PRIME_DELAY)
        self._activity_callbacks.append(callback)

        def unregister_activity_callback() -> None:
            self._activity_callbacks.remove(callback)

        return unregister_activity_callback

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
            op_response_callback=self._hold_update,
        )

    def _disconnected_callback(self) -> None:
        """Handle a disconnect from the lock."""
        _LOGGER.debug("%s: Disconnected from lock callback", self.name)
        if self._update_in_flight():
            # The running update re-arms the reconnect when it ends.
            return
        self._schedule_reconnect()

    def _update_in_flight(self) -> bool:
        return self._update_task is not None and not self._update_task.done()

    def _schedule_reconnect(self) -> None:
        """Reconnect an always-connected lock, paced by the backoff."""
        if (
            self._running
            and self._always_connected
            and not _AUTH_FAILURE_HISTORY.should_raise(self.address)
        ):
            _LOGGER.debug("%s: Scheduling reconnect", self.name)
            self._schedule_future_update_with_debounce(self._reconnect_backoff)

    def _keep_alive(self) -> None:
        """Keep the lock connection alive."""
        if not self._always_connected:
            return
        _LOGGER.debug("%s: Executing keep alive", self.name)
        if not self._update_in_flight():
            if self.is_connected and self._floor_outlasts_the_link():
                # A read that returns no position keeps the link up when the
                # floor's end would come after the lock's idle cutoff.
                self.background_task(self._keep_link_alive())
            # Debounced so a backoff longer than KEEP_ALIVE_TIME is not pushed
            # out on every tick; a running update re-arms on its own exit.
            self._schedule_future_update_with_debounce(self._reconnect_backoff)
        self._schedule_next_keep_alive(KEEP_ALIVE_TIME)

    async def _keep_link_alive(self) -> None:
        """Send a read that returns no position: the battery, or else the door."""
        async with self._operation_lock:
            if (
                not self._running
                or not self.is_connected
                or not self._floor_outlasts_the_link()
            ):
                return
            # Connected, so this hands back the link without a connect.
            lock = await self._ensure_connected()
            info = self._lock_info
            if info is None:
                return
            try:
                asked = await self._poll_battery(lock, force=True)
                if not asked and info.door_sense:
                    await lock.door_status()
                    asked = True
            except (TimeoutError, *RETRYABLE_EXCEPTIONS) as ex:
                # The cycle at the floor's end retries and reports the link.
                _LOGGER.debug("%s: Keep-alive read failed: %s", self.name, ex)
                return
            if asked:
                # A request, so the floor check and the keep-alive count from it.
                self._last_operation_complete_time = time.monotonic()
                self._reschedule_next_keep_alive()

    def _floor_outlasts_the_link(self) -> bool:
        """Whether the cycle at the floor's end would come after the idle cutoff."""
        # A second of margin for the request to reach the lock.
        return (
            self._earliest_update_time - self._last_operation_complete_time
            > IDLE_DISCONNECT_TIME - 1.0
        )

    def _clear_reconnect_backoff(self) -> None:
        """A completed update or operation proves the link; drop the backoff."""
        self._reconnect_backoff = 0.0

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
        """Set securemode; raises OperationFailedError on a reported failure."""
        await self._run_lock_operation(_SECUREMODE_OPERATION)

    async def lock(self) -> None:
        """Lock the lock; raises OperationFailedError on a reported failure."""
        await self._run_lock_operation(_LOCK_OPERATION)

    async def unlock(self) -> None:
        """Unlock the lock; raises OperationFailedError on a reported failure."""
        await self._run_lock_operation(_UNLOCK_OPERATION)

    async def unlatch(self) -> None:
        """Unlatch (momentarily open) the lock.

        Completes as UNLATCHED; the state the lock settles to after the dwell
        arrives as a later status update. Raises OperationFailedError on a
        reported failure.
        """
        await self._run_lock_operation(_UNLATCH_OPERATION)

    def _init_operation_state(self) -> None:
        """Initialize the per-operation fields so they read before any operation."""
        self._operation_outcome = None
        self._operation_window_open = False
        self._operation_in_flight = False
        self._operation_issued_at = NEVER_TIME
        self._operation_answered = False
        self._operation_failure = None

    @operation_lock
    async def _run_lock_operation(self, operation: _LockOperation) -> None:
        """Run a lock operation; _finalize_operation runs on every exit."""
        self._cancel_future_update()
        self._operation_outcome = None
        self._operation_answered = False
        self._operation_failure = None
        self._operation_in_flight = True
        # Taken under the operation lock, so a command queued behind another
        # counts from when it runs.
        self._operation_issued_at = time.monotonic()
        try:
            await self._execute_lock_operation(operation)
        except Exception:
            self._operation_outcome = LockStatus.UNKNOWN
            raise
        finally:
            self._finalize_operation()

    def _jam_takes_precedence(self) -> bool:
        """Whether a reported jam or setup condition takes precedence over the command.

        True for a report within JAMMED_PRECEDENCE_TIME before the command was
        issued, or since.
        """
        return (
            self._last_jam_event_time
            >= self._operation_issued_at - JAMMED_PRECEDENCE_TIME
        )

    def _end_a_retry_under_precedence(self, failure: Exception | None) -> None:
        """Refuse to retry a failed attempt while a reported jam takes precedence."""
        if failure is not None and self._jam_takes_precedence():
            raise OperationIncompleteError(
                f"{self.name}: a jam or setup condition the lock reported "
                "takes precedence over the command, whose attempt ended with "
                f"{failure!r}; it was not retried"
            ) from failure

    def _attention_holds_display(self, state: LockStatus, window_live: bool) -> bool:
        """Whether an attention status on display keeps state off it while window_live.

        Logs the state it keeps off the display.
        """
        if not (window_live and self.lock_status in MANUAL_INTERVENTION_STATUSES):
            return False
        _LOGGER.debug(
            "%s: %s not applied; a status needing attention holds the display",
            self.name,
            state,
        )
        return True

    def _operation_result(self, operation: _LockOperation, succeeded: bool) -> None:
        """Close the operation window and display a successful op-response.

        A failure's JAMMED reached the display on the same frame.
        """
        self._operation_answered = True
        self._close_operation_window()
        if not self._running or not succeeded:
            return
        if self._attention_holds_display(
            operation.complete, self._jam_takes_precedence()
        ):
            return
        # The answer supersedes a held status.
        self._release_jam_hold()
        self._display(
            _project_lock_status(operation.complete, self._get_current_state())
        )

    def _operation_write_success(self, operation: _LockOperation) -> None:
        """Display the operation's transitional state, releasing any display hold.

        Skipped once the op-response has been handled, and while an attention
        status holds the display.
        """
        if self._operation_answered:
            return
        pending = operation.pending(self._get_current_state())
        if self._attention_holds_display(pending.lock, self._jam_takes_precedence()):
            return
        self._release_jam_hold()
        self._display(pending)

    def _close_operation_window(self) -> None:
        self._operation_window_open = False

    def _finalize_operation(self) -> None:
        """Close the operation window, display the outcome, schedule the next poll."""
        self._operation_in_flight = False
        # The op-response decided the display when it arrived; the exit
        # decides it only for an operation that ended without one.
        outcome = None if self._operation_answered else self._operation_outcome
        self._close_operation_window()
        # Set before the stop check so a restarted watcher inherits them.
        self._force_lock_status_poll = True
        self._hold_update()
        if not self._running:
            # Stopped mid-operation: a cycle armed now would outlive the stop.
            # A restarted watcher polls the lock afresh, so a report from before
            # the stop neither takes precedence over its commands nor holds its
            # display.
            self._reset_jam_state()
            return
        # A hold still live here means the command's write never succeeded.
        if outcome is not None and not self._attention_holds_display(
            outcome, self._jam_takes_precedence() or self._jam_hold_live()
        ):
            self._display(_project_lock_status(outcome, self._get_current_state()))
        # The exit owns the next poll; drop any cycle armed during the operation.
        self._cancel_future_update()
        if self._always_connected and not self.is_connected:
            # This cycle is the dropped link's reconnect, so a hold cannot skip it.
            self._schedule_future_update_with_debounce(LOCK_STALE_STATE_DEBOUNCE_DELAY)
        elif not self._jam_hold_live():
            # A live hold polls at its own end. Unsettled: poll once the motor
            # has stopped, not at the keep-alive.
            if _Display(self.lock_status, self.secure_status).settled:
                delay = KEEP_ALIVE_TIME
            else:
                delay = LOCK_STALE_STATE_DEBOUNCE_DELAY
            self._schedule_future_update_with_debounce(delay)

    def _init_jam_state(self) -> None:
        """Initialize the jam state."""
        self._jammed_hold_deadline = NEVER_TIME
        self._jam_hold_timer: asyncio.TimerHandle | None = None
        # When the lock last reported a jam or setup condition.
        self._last_jam_event_time = NEVER_TIME

    def _reset_jam_state(self) -> None:
        """Forget the reported jam: a restarted watcher polls the lock afresh."""
        self._last_jam_event_time = NEVER_TIME
        self._release_jam_hold()

    def _jam_hold_live(self) -> bool:
        """Whether a reported jam or setup condition is within its display hold."""
        return time.monotonic() < self._jammed_hold_deadline

    def _schedule_jam_hold_timer(self, delay: float) -> None:
        self._cancel_jam_hold_timer()
        self._jam_hold_timer = self.loop.call_later(delay, self._jam_hold_ended)

    def _cancel_jam_hold_timer(self) -> None:
        if self._jam_hold_timer:
            self._jam_hold_timer.cancel()
            self._jam_hold_timer = None

    def _release_jam_hold(self) -> None:
        if self._jammed_hold_deadline != NEVER_TIME:
            _LOGGER.debug("%s: Releasing the display hold", self.name)
        self._jammed_hold_deadline = NEVER_TIME
        self._cancel_jam_hold_timer()

    def _jam_hold_ended(self) -> None:
        """Poll the lock now that the display hold has ended."""
        self._jam_hold_timer = None
        if self._operation_lock.locked():
            # The operation's exit would clear a cycle armed now, so retry.
            self._schedule_jam_hold_timer(OPERATION_IN_PROGRESS_DEFER_SECONDS)
            return
        # Forced: the held status already counts as this session's reading.
        self._force_lock_status_poll = True
        self._schedule_future_update_with_debounce(0)

    def _admit_lock_status(self, incoming: LockStatus) -> LockStatus | None:
        """Decide the displayed lock status for a value the lock reported.

        None refuses the value, so nothing from it is applied.
        """
        if incoming in TRANSITIONAL_READINGS:
            # A lock polled mid-motion answers with the position it is leaving.
            self._hold_update()
        if incoming in MANUAL_INTERVENTION_STATUSES:
            now = time.monotonic()
            self._last_jam_event_time = now
            if (
                self.lock_status not in MANUAL_INTERVENTION_STATUSES
                or self._operation_in_flight
            ):
                # Arm on the transition, or on any report during our command;
                # re-arming on repeats would poll a demand-connected lock forever.
                _LOGGER.debug(
                    "%s: Holding %s on display for %s seconds",
                    self.name,
                    incoming,
                    JAMMED_HOLD_TIME,
                )
                self._jammed_hold_deadline = now + JAMMED_HOLD_TIME
                self._schedule_jam_hold_timer(JAMMED_HOLD_TIME)
            return incoming
        if self._operation_window_open:
            # The operation applies its own outcome, so a reading inside its
            # window is not shown ahead of it.
            _LOGGER.debug(
                "%s: Operation window open, not accepting lock status %s",
                self.name,
                incoming,
            )
            return None
        if self._attention_holds_display(incoming, self._jam_hold_live()):
            return None
        return incoming

    # Only called from _run_lock_operation, which holds the operation lock.
    @retry_bluetooth_connection_error(lock_operation=True)
    async def _execute_lock_operation(self, operation: _LockOperation) -> None:
        """Execute a lock operation."""
        if not self._running:
            raise RuntimeError(
                f"{self.name}: Lock operation not possible because not running"
            )
        op_attr = operation.op_attr
        _LOGGER.debug("%s: Starting %s", self.name, op_attr)
        try:
            lock = await self._ensure_connected()
            self._end_a_retry_under_precedence(self._operation_failure)
            self._cancel_future_update()
            # Opened before the write: a status handled while the write is
            # awaited cannot be placed before or after the lock took the command.
            self._operation_window_open = True
            await getattr(lock, op_attr)(
                write_success_callback=functools.partial(
                    self._operation_write_success, operation
                ),
                result_callback=functools.partial(self._operation_result, operation),
            )
        except OperationFailedError:
            # The lock answered, so the exit moves the timers as a success does;
            # its JAMMED is already on display.
            self._complete_operation(time.monotonic())
            raise
        except Exception as ex:
            self._operation_failure = ex
            # Closed so a status the lock reports before the retry is admitted.
            self._close_operation_window()
            _LOGGER.debug(
                "%s: Failed to execute lock operation due to %s",
                self.name,
                ex,
            )
            raise
        _LOGGER.debug("%s: Finished %s", self.name, op_attr)
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

    async def get_keycode(self, slot: int) -> str | None:
        """Read the PIN in a keypad slot; None if the slot is empty."""
        validate_keycode_slot(slot)
        return await self._run_on_lock(
            "get_keycode", lambda lock: lock.get_keycode(slot)
        )

    async def set_keycode(self, slot: int, pin: str) -> None:
        """Program a PIN into a keypad slot.

        The slot is cleared first, so any failure after the Clear can leave it empty.
        """
        validate_keycode_slot(slot)
        encode_keycode_pin(pin)
        await self._run_on_lock("set_keycode", lambda lock: lock.set_keycode(slot, pin))

    async def clear_keycode(self, slot: int) -> None:
        """Clear a keypad slot."""
        validate_keycode_slot(slot)
        await self._run_on_lock("clear_keycode", lambda lock: lock.clear_keycode(slot))

    @operation_lock
    async def _run_on_lock(
        self, name: str, operation: Callable[[Lock], Awaitable[_T]]
    ) -> _T:
        """Run a command that moves no motor; a failure is logged once at warning."""
        if not self._running:
            raise RuntimeError(f"{self.name}: {name} not possible because not running")
        try:
            outcome = await self._attempt_on_lock(name, operation)
            if isinstance(outcome, TimeoutError):
                raise TimeoutError(  # noqa: TRY301
                    f"{self.name}: lock did not answer {name}"
                ) from outcome
        except Exception as err:
            # A ResponseError's message can hold the rejected frame's hex.
            detail = "" if isinstance(err, ResponseError) else f": {err}"
            _LOGGER.warning(
                "%s: %s failed: %s%s", self.name, name, type(err).__name__, detail
            )
            raise
        return outcome

    @retry_bluetooth_connection_error
    async def _attempt_on_lock(
        self, name: str, operation: Callable[[Lock], Awaitable[_T]]
    ) -> _T | TimeoutError:
        """Run one attempt; a TimeoutError from the command's wait is returned."""
        # Not reconnected to resend after a stop.
        if not self._running:
            raise RuntimeError(
                f"{self.name}: stopped during {name}; its result is unknown"
            )
        lock = await self._ensure_connected()
        try:
            result = await operation(lock)
        except TimeoutError as err:
            # Raised, it would make the retry decorator reconnect and resend.
            return err
        self._record_auth_success()
        self._complete_operation(time.monotonic())
        return result

    def _complete_operation(self, now: float) -> None:
        """Mark an operation as complete and reset timers."""
        self._last_operation_complete_time = now
        self._clear_reconnect_backoff()
        if self._activity_callbacks and not self._activity_primed:
            # A lock operation cancelled any pending update, the priming one included.
            self._schedule_future_update_with_debounce(ACTIVITY_PRIME_DELAY)
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
            self.secure_status,
        )

    def _stage_display(
        self, lock_state: LockState, display: _Display, changes: dict[str, Any]
    ) -> None:
        """Stage the lock pair of display into changes and record whether it settled."""
        if display.settled:
            self._seen_this_session.add(LockStatus)
        else:
            # An unsettled display must not suppress the follow-up poll.
            self._seen_this_session.discard(LockStatus)
        if lock_state.lock != display.lock:
            changes["lock"] = display.lock
            self._activity_drain_pending = True
        if lock_state.secure != display.secure:
            changes["secure"] = display.secure

    def _display(self, display: _Display) -> None:
        """Put an operation's display up; it is not a report, so no admission."""
        _LOGGER.debug("%s: Displaying %s", self.name, display)
        lock_state = self._get_current_state()
        changes: dict[str, Any] = {}
        self._stage_display(lock_state, display, changes)
        self._apply_changes(lock_state, changes)

    def _update_any_state(self, states: Iterable[LockStateValue | AuthState]) -> None:
        """Apply states the lock reported to the display."""
        _LOGGER.debug("%s: State changed: %s", self.name, states)
        lock_state = self._get_current_state()
        changes: dict[str, Any] = {}
        for state in states:
            if isinstance(state, BatteryState) and state.voltage <= 3.0:
                # A refused reading must not stand as seen, so _poll_battery
                # can ask again; the cooldown paces that next ask.
                self._seen_this_session.discard(BatteryState)
                self._earliest_battery_attempt_time = (
                    time.monotonic() + BATTERY_TIMEOUT_COOLDOWN
                )
                _LOGGER.warning(
                    "%s: Battery voltage is impossible: %s; "
                    "not asking again for %d seconds",
                    self.name,
                    state.voltage,
                    BATTERY_TIMEOUT_COOLDOWN,
                )
                continue
            self._seen_this_session.add(type(state))
            if isinstance(state, AuthState):
                if lock_state.auth != state:
                    changes["auth"] = state
            elif isinstance(state, LockStatus):
                # Every reported lock status, repeats included, is admitted here.
                admitted = self._admit_lock_status(state)
                if admitted is None:
                    # A refused reading must not suppress the follow-up poll.
                    self._seen_this_session.discard(LockStatus)
                    continue
                if admitted in SETUP_CONDITION_STATUSES and lock_state.lock != admitted:
                    _LOGGER.warning(
                        "%s: Lock reports %s, a setup condition that ends "
                        "at the lock by hand",
                        self.name,
                        admitted,
                    )
                self._stage_display(
                    lock_state, _project_lock_status(admitted, lock_state), changes
                )
            elif isinstance(state, DoorStatus):
                if lock_state.door != state:
                    changes["door"] = state
                    self._activity_drain_pending = True
            elif isinstance(state, BatteryState):
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
        self._apply_changes(lock_state, changes)

    def _apply_changes(self, lock_state: LockState, changes: dict[str, Any]) -> None:
        """Publish lock_state with changes applied, if any."""
        if not changes:
            return
        self._callback_state(replace(lock_state, **changes))

    def _record_auth_success(self) -> None:
        """Clear the auth failure count and publish AuthState(successful=True)."""
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
        self._clear_reconnect_backoff()
        _LOGGER.debug("%s: Finished validate", self.name)

    async def _poll_battery(self, lock: Lock, *, force: bool = False) -> bool:
        """Poll battery if needed: periodic refresh, cooldown, errors.

        Battery state requires a poll of the lock to update. In always_connected mode
        _seen_this_session never clears, so once the refresh deadline passes
        BatteryState is evicted to force a re-poll -- but only after the cooldown gate.
        force asks even when the reading is current.

        Returns True if the lock was asked, whether or not it answered.
        """
        assert self._lock_info is not None  # nosec
        if not self._lock_info.battery_reporting:
            _LOGGER.debug(
                "%s: Needs battery workaround model %s",
                self.name,
                self._lock_info.model,
            )
            return False

        now = time.monotonic()
        # Skip while in cooldown, after a read the lock did not answer or a
        # reading that was thrown away.
        if now < self._earliest_battery_attempt_time:
            _LOGGER.debug(
                "%s: Skipping battery request; not asking again for %d seconds",
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
        if not force and BatteryState in self._seen_this_session:
            return False

        try:
            await lock.battery()
            self._record_auth_success()
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
    async def _update(self) -> None:
        """Run one update cycle under the operation lock."""
        await self._locked_update()

    @retry_bluetooth_connection_error
    async def _locked_update(self) -> None:
        """Update the lock state.

        Returns nothing. Every value this cycle asks for is applied as the
        lock's answering frame arrives, so a returned state would be a second
        reading, taken later than the one the callback already delivered. A
        caller takes the state from the callback or from the properties.
        """
        has_lock_info = self._lock_info is not None

        _LOGGER.debug(
            "%s: Starting update (has_lock_info: %s)", self.name, has_lock_info
        )
        lock = await self._ensure_connected()
        if not self._lock_info:
            self._lock_info = await self._probe_lock_info(lock)

        # The reads below are issued here, and _update_any_state processes each
        # answer, so the returned values are not used.
        # Asking for battery first seems to reduce the chance of the lock
        # getting into a bad state.
        made_request = await self._poll_battery(lock)

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
        # A post-operation poll asks regardless; the seen reading may be stale.
        if (
            self._force_lock_status_poll
            or LockStatus not in self._seen_this_session
            or (
                not made_request
                and self._always_connected
                and not self._activity_drain_due()
            )
        ):
            made_request = True
            await lock.lock_status()
            # Cleared only once a poll answered, so a failed poll retries.
            self._force_lock_status_poll = False
            self._record_auth_success()

        _LOGGER.debug("%s: Finished update", self.name)

        current = self._get_current_state()
        # Notify consumers that the update is complete, even if nothing changed.
        self._callback_state(current)

        if await self._poll_activity(lock, first_update=not has_lock_info):
            made_request = True

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

    def _activity_drain_due(self) -> bool:
        """True when a registered consumer is owed a drain and no backoff holds it."""
        return bool(
            self._activity_callbacks
            and self._activity_drain_pending
            and time.monotonic() >= self._earliest_activity_drain_time
        )

    async def _poll_activity(self, lock: Lock, *, first_update: bool) -> bool:
        """Drain the activity log if one is due; True if the lock was asked."""
        if not self._activity_drain_due():
            return False
        if first_update:
            self._schedule_future_update_with_debounce(ACTIVITY_PRIME_DELAY)
            return False
        await self._drain_activity(lock)
        return True

    async def _drain_activity(self, lock: Lock) -> None:
        """Read the lock's activity log and deliver the records.

        The first drain only primes, discarding the backlog. An incomplete
        drain stays pending; records already read are delivered regardless.
        """
        # Consumed before the await so a change during the drain is kept.
        self._activity_drain_pending = False
        records = 0
        try:
            async for activity in lock.drain_lock_activity():
                records += 1
                if self._activity_primed:
                    self._callback_activity(activity)
        except BaseException as err:
            self._activity_drain_pending = True
            if not isinstance(err, (BleakError, TimeoutError, ResponseError)):
                raise
            self._note_activity_drain_failure(err, records)
            return
        self._activity_drain_failures = 0
        self._activity_overruns = 0
        if not self._activity_primed:
            _LOGGER.debug(
                "%s: Primed the activity log, discarding %d backlog records",
                self.name,
                records,
            )
        self._activity_primed = True

    def _note_activity_drain_failure(self, err: Exception, records: int) -> None:
        """Count a failed drain; after enough in a row, leave the log alone a while."""
        if isinstance(err, ActivityLogOverrunError):
            self._activity_overruns += 1
            if records and self._activity_overruns < ACTIVITY_DRAIN_MAX_OVERRUNS:
                # A deep backlog, not a failure: carry on shortly.
                _LOGGER.debug(
                    "%s: Activity log is deeper than one drain, continuing shortly",
                    self.name,
                )
                self._schedule_future_update_with_debounce(ACTIVITY_PRIME_DELAY)
                return
            # Nothing read, or no end after many chunks: a log that never ends.
            self._activity_overruns = 0
            strikes = ACTIVITY_DRAIN_FAILURE_THRESHOLD
        else:
            strikes = 1
        self._activity_drain_failures += strikes
        if self._activity_drain_failures < ACTIVITY_DRAIN_FAILURE_THRESHOLD:
            _LOGGER.debug(
                "%s: Reading lock activity failed (%s), will retry on next update.",
                self.name,
                err,
            )
            return
        self._activity_drain_failures = 0
        self._earliest_activity_drain_time = (
            time.monotonic() + ACTIVITY_DRAIN_FAILURE_BACKOFF
        )
        _LOGGER.warning(
            "%s: Reading lock activity keeps failing (%s); "
            "not trying again for %d seconds",
            self.name,
            err,
            ACTIVITY_DRAIN_FAILURE_BACKOFF,
        )

    def _callback_activity(self, activity: ActivityValue) -> None:
        """Call the activity callbacks."""
        for callback in self._activity_callbacks:
            try:
                callback(activity)
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("%s: Error calling activity callback", self.name)

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
            _LOGGER.debug("%s: Requested slow connection parameters", self.name)

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
        connection_info = self.connection_info
        if self._lock_info is None or connection_info is None:
            # Kept until both are known; the debug line above shows which is not.
            return
        for callback in self._callbacks:
            try:
                callback(lock_state, self._lock_info, connection_info)
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("%s: Error calling callback", self.name)

    def _read_yale_advertisement(self, mfr_data: Mapping[int, bytes]) -> float:
        """Track the Yale advertised value; return the update delay it calls for."""
        # YALE_MFR_ID advertisements come in two formats, and a payload under
        # ASSA_ABLOY_MFR_ID is read the same way:
        # - 1-byte: activity flag (0/1), used for change detection
        # - 18-byte: 2 header bytes + the lock's 16-byte cloud ID (the
        #   identifier used by the Yale/ASSA ABLOY cloud API), e.g.
        #   b'\x00\x00\x80\x15\xd0\x11\xf7\xa5\x43\x1f\x85\xd7\xff\x23\x5f\x1e\x75\x46'
        # Only track byte[0] from the 1-byte format. The two formats
        # alternate every few seconds; without the length check, the
        # static 0x00 header of the 18-byte format causes repeated
        # connections if it differs from the 1-byte value.
        is_first_advertisement = self._last_adv_value == -1
        yale_mfr_id, yale_data = _select_yale_payload(mfr_data)
        next_update = 0.0
        # An empty payload is skipped rather than indexed.
        if yale_data and (len(yale_data) == 1 or is_first_advertisement):
            current_value = yale_data[0]
            if is_first_advertisement:
                # We haven't seen a valid value yet so we schedule an update
                next_update = FIRST_UPDATE_COALESCE_SECONDS
            elif (
                current_value in VALID_ADV_VALUES
                and current_value != self._last_adv_value
            ):
                # The flag also flips for activity with no lock or door change,
                # such as a doorbell press, which only the activity log records.
                self._activity_drain_pending = True
                next_update = ADV_UPDATE_COALESCE_SECONDS
            self._last_adv_value = current_value
        elif yale_data is not None and not yale_data:
            _LOGGER.debug(
                "%s: Empty Yale advertisement payload under company ID %s; skipped",
                self.name,
                yale_mfr_id,
            )
        return next_update

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
        # radio input and its length is not ours to assume. Each refusal is
        # logged at debug — not INFO like the notify gate — because
        # advertisements repeat every few seconds, so a chronic condition
        # would flood any stronger level; debug keeps it diagnosable.
        if apple_data := mfr_data.get(APPLE_MFR_ID):
            first_byte = apple_data[0]
            if first_byte == HAP_FIRST_BYTE:
                if (hk_state := get_homekit_state_num(apple_data)) is None:
                    _LOGGER.debug(
                        "%s: %d-byte HomeKit advertisement ends before its"
                        " state record; skipped",
                        self.name,
                        len(apple_data),
                    )
                else:
                    # Sometimes the yale data is glued on to the end of the
                    # HomeKit data but in that case it seems wrong so we
                    # don't process it
                    #
                    # if len(mfr_data[APPLE_MFR_ID]) > 20
                    #     and YALE_MFR_ID not in mfr_data:
                    # mfr_data[YALE_MFR_ID] = mfr_data[APPLE_MFR_ID][20:]
                    if self._last_hk_state == -1:
                        # We haven't seen a HomeKit state yet so we schedule
                        # an update
                        next_update = FIRST_UPDATE_COALESCE_SECONDS
                    elif hk_state != self._last_hk_state:
                        next_update = HK_UPDATE_COALESCE_SECONDS
                    self._last_hk_state = hk_state
            elif first_byte == HAP_ENCRYPTED_FIRST_BYTE:
                # Encrypted data, we don't know how to decrypt it
                # but we know its a state change so we schedule an update
                next_update = HK_UPDATE_COALESCE_SECONDS
        elif APPLE_MFR_ID in mfr_data:
            _LOGGER.debug("%s: Empty HomeKit advertisement payload; skipped", self.name)
        yale_update = self._read_yale_advertisement(mfr_data)
        next_update = next_update or yale_update
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
        self._clear_reconnect_backoff()
        self._first_update_future = asyncio.get_running_loop().create_future()
        if device := await get_device(self.address):
            self.set_ble_device(device)
            self._schedule_future_update_with_debounce(ADV_UPDATE_COALESCE_SECONDS)

        return self._cancel

    def _cancel(self) -> None:
        self._running = False
        self._cancel_future_update()
        if not self._operation_in_flight:
            # Kept while a command is in flight so a jam reported during it
            # still ends its attempts; the stopped exit resets it instead.
            self._reset_jam_state()
        self.background_task(self._execute_forced_disconnect("stopping"))

    def background_task(self, fut: Coroutine[Any, Any, Any]) -> None:
        """Execute a background task."""
        task: asyncio.Task[Any] = asyncio.create_task(fut)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        """Remove finished task and log unexpected exceptions."""
        self._background_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            _LOGGER.error(
                "%s: Background task failed: %s", self.name, exc, exc_info=exc
            )

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

    def _hold_update(self) -> None:
        """Hold update cycles off for the stale-state debounce from now."""
        self._earliest_update_time = time.monotonic() + LOCK_STALE_STATE_DEBOUNCE_DELAY

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
        """Schedule an update in future seconds, never before _earliest_update_time."""
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

    def _wait_for_the_floor(self, now: float) -> bool:
        """Return True while the update floor holds; re-arm unless a timer stands."""
        if now >= self._earliest_update_time:
            return False
        if self._cancel_deferred_update is None:
            _LOGGER.debug("%s: Rescheduling update to avoid stale state", self.name)
            self._schedule_future_update(self._earliest_update_time - now)
        return True

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
        if self._wait_for_the_floor(now):
            return
        if self._operation_lock.locked():
            # The cycle is re-armed rather than created, so no task sits on the
            # operation lock waiting for the motor to stop.
            _LOGGER.debug(
                "%s: Rescheduling update until the operation lock is released",
                self.name,
            )
            self._schedule_future_update_with_debounce(
                OPERATION_IN_PROGRESS_DEFER_SECONDS
            )
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

    async def _execute_deferred_update(self) -> None:  # noqa: PLR0915
        """Execute deferred update."""
        _LOGGER.debug("%s: Deferred update starting", self.name)
        if not self._running:
            _LOGGER.debug("%s: Deferred updated ignored because not running", self.name)
            return
        _LOGGER.debug("%s: Starting deferred update", self.name)
        failed = True
        cancelled = False
        yielded = False
        try:
            async with self._operation_lock:
                # Re-check: an operation may have run since the timer fired.
                if self._wait_for_the_floor(time.monotonic()):
                    yielded = True  # No update ran, so there is no outcome to record.
                    return
                await self._locked_update()
            failed = False
            self._set_update_state(None)
        except AuthError as ex:
            self._set_update_state(ex)
            _LOGGER.exception(
                "%s: Auth error: key or slot (key index) is incorrect",
                self.name,
            )
        except asyncio.CancelledError:
            cancelled = True  # this library giving up, not the lock failing
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
        finally:
            if not cancelled and not yielded:
                self._record_update_outcome(failed)
            if not self.is_connected:
                # The disconnect callback stood down while this update ran.
                self._schedule_reconnect()

    def _record_update_outcome(self, failed: bool) -> None:
        """Double the reconnect backoff on a failed update; clear it on success."""
        if not failed:
            self._clear_reconnect_backoff()
            return
        self._reconnect_backoff = min(
            MAX_RECONNECT_BACKOFF_TIME,
            self._reconnect_backoff * 2 or RECONNECT_BACKOFF_TIME,
        )


# The HomeKit state record inside the advertisement payload: acid, the global
# state number, cn, cv, starting at byte 9.
_HAP_STATE_RECORD = struct.Struct("<HHBB")
_HAP_STATE_RECORD_OFFSET = 9


def get_homekit_state_num(data: bytes) -> int | None:
    """Get the homekit state number from the manufacturer data.

    Returns None when the payload ends before the record does: the
    advertisement is radio input and its length is not ours to assume.
    """
    if len(data) < _HAP_STATE_RECORD_OFFSET + _HAP_STATE_RECORD.size:
        return None
    _acid, gsn, _cn, _cv = _HAP_STATE_RECORD.unpack_from(data, _HAP_STATE_RECORD_OFFSET)
    return gsn
