from __future__ import annotations

import asyncio
import bisect
import logging
import os
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, TypeVar, cast

from bleak import BleakError
from bleak_retry_connector import (
    MAX_CONNECT_ATTEMPTS,
    BleakClientWithServiceCache,
    BLEDevice,
    establish_connection,
)

from . import util
from .const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
    MECHANICAL_OPERATION_ERRORS,
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    VALUE_TO_DOOR_STATUS,
    VALUE_TO_LOCK_OPERATION_REMOTE_TYPE,
    VALUE_TO_LOCK_OPERATION_SOURCE,
    VALUE_TO_LOCK_STATUS,
    VALUE_TO_OPERATION_ERROR,
    AutoLockMode,
    AutoLockState,
    BatteryState,
    Commands,
    DoorActivity,
    DoorStatus,
    LockActivity,
    LockActivityType,
    LockInfo,
    LockOperationRemoteType,
    LockOperationSource,
    LockStateValue,
    LockStatus,
    OperationError,
    SettingType,
    StatusType,
)
from .secure_session import SecureSession
from .session import (
    OPERATION_RESPONSE_TIMEOUT,
    UNLATCH_OPERATION_RESPONSE_TIMEOUT,
    AuthError,
    DisconnectedError,
    OperationFailedError,
    OperationIncompleteError,
    OperationProgress,
    Session,
    UnlatchError,
    YaleXSBLEError,
)

_LOGGER = logging.getLogger(__name__)

LOCK_INFO_TIMEOUT = 3

# byte[4] of an Unlock command: 0x0A turns the plain unlock into an unlatch
# (retract the latch). Same opcode+optype mechanism securemode uses on the
# Lock opcode (0x04).
UNLATCH_OPERATION_BYTE = 0x0A
SECUREMODE_OPERATION_BYTE = 0x04

AA_BATTERY_VOLTAGE_TO_PERCENTAGE = (
    (1.55, 100),
    (1.549, 97),
    (1.548, 95),
    (1.547, 94),  # confirmed
    (1.49075, 93),  # confirmed
    (1.49, 90),
    (1.471, 85),  # confirmed
    (1.46, 80),
    (1.45, 75),
    (1.40, 70),
    (1.39, 65),
    (1.38, 60),
    (1.37, 55),
    (1.36, 50),
    (1.35, 45),
    (1.34, 40),
    (1.33, 35),
    (1.32, 30),
    (1.31, 28),
    (1.30, 27),
    (1.29, 25),
    (1.28, 20),
    (1.27, 15),
    (1.26, 10),
    (1.25, 5),
    (1.24, 0),
)
AA_BATTERY_VOLTAGE_LIST = [
    voltage for voltage, _ in sorted(AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
]
AA_BATTERY_VOLTAGE_MAP = dict(AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
WrapFuncType = TypeVar("WrapFuncType", bound=Callable[..., Any])


def raise_if_not_connected(func: WrapFuncType) -> WrapFuncType:
    """Define a wrapper to raise if we are not connected to the lock."""

    async def _async_wrap_connected_operation(
        self: Lock, *args: Any, **kwargs: Any
    ) -> None:
        """Wrap a function to make sure the lock is connected."""
        if not self.is_connected:
            raise DisconnectedError("Lock is not connected")
        return await func(self, *args, **kwargs)

    return cast(WrapFuncType, _async_wrap_connected_operation)


def convert_voltage_to_percentage(voltage: float) -> int:
    """Convert voltage to percentage."""
    pos = bisect.bisect_left(AA_BATTERY_VOLTAGE_LIST, voltage)
    if pos != 0:
        pos -= 1
    return AA_BATTERY_VOLTAGE_MAP[AA_BATTERY_VOLTAGE_LIST[pos]]


def _settings_response_matcher(
    command_value: int, setting_value: int
) -> Callable[[bytes], bool]:
    """Match the 0xBB settings frame answering one command/setting pair.

    A settings command (READSETTING/WRITESETTING) is answered by two frames:
    an 0xAA acknowledgment echoing the request, then the 0xBB frame carrying
    the stored value. The acknowledgment's value field is zero, so completing
    the solicited wait on it misreads every setting; the wait must hold out
    for the 0xBB frame that echoes the command opcode and the setting id.
    """

    def matches(data: bytes) -> bool:
        return (
            len(data) >= 0x0C
            and data[0x00] == 0xBB
            and data[0x01] == command_value
            and data[0x04] == setting_value
        )

    return matches


def _describe_operation_error(result: int) -> str:
    """Name an op-response result byte for a reader.

    The log record of a failure and the exception raised for the same frame
    have to name the same cause, so both take the name from here rather than
    decoding the byte separately.
    """
    error = VALUE_TO_OPERATION_ERROR.get(result)
    return error.name if error else "unknown"


def _ack_matcher(opcode: int, operation_byte: int) -> Callable[[bytes], bool]:
    """Match the acknowledgement of the written command.

    The acknowledgement echoes the request: 0xAA, the written opcode in
    byte[1], and the command's operation byte in byte[4] (a plain operation
    carries 0x00 there, securemode 0x04, unlatch 0x0A). Any other frame is
    not this command's acknowledgement; it stays on the state-callback path
    and the wait continues.

    The 0x00 and 0x04 echoes are field captured. The 0x0A echo is inferred
    from the same rule, since no unlatch has been captured, and a wrong
    inference costs nothing: an unlatch does not wait on the acknowledgement
    stage, so a lock answering it with a zero byte[4] leaves the operation
    unacknowledged and nothing more.
    """

    def _matches(data: bytes) -> bool:
        return (
            len(data) > 0x04
            and data[0] == 0xAA
            and data[1] == opcode
            and data[4] == operation_byte
        )

    return _matches


def _operation_response_matcher(opcode: int) -> Callable[[bytes], bool]:
    """Match the op-response that completes a mechanical operation.

    The op-response (0xBB + the sent opcode) is emitted when the motor stops
    and carries the operation result in byte[15].
    """

    def _matches(data: bytes) -> bool:
        return len(data) > 0x0F and data[0] == 0xBB and data[1] == opcode

    return _matches


class Lock:
    def __init__(
        self,
        ble_device_callback: Callable[[], BLEDevice],
        keyString: str,
        keyIndex: int,
        name: str,
        state_callback: Callable[[Iterable[LockStateValue]], None],
        info: LockInfo | None = None,
        disconnect_callback: Callable[[], None] | None = None,
        ack_callback: Callable[[], None] | None = None,
        op_response_callback: Callable[[], None] | None = None,
    ) -> None:
        self.ble_device_callback = ble_device_callback
        self.key = bytes.fromhex(keyString)
        self.key_index = keyIndex
        self.name = name
        self.session: Session | None = None
        self.secure_session: SecureSession | None = None
        self.is_secure = False
        self._lock = asyncio.Lock()
        self._lock_info = info
        self.client: BleakClientWithServiceCache | None = None
        self._state_callback = state_callback
        # Invoked when an operation acknowledgement (0xAA echoing LOCK/UNLOCK) is
        # parsed, so the consumer can timestamp it. The acknowledgement is consumed
        # mid-wait by the session's typed matcher and never surfaces as a return
        # value, so this hook is the only way to observe it.
        self._ack_callback = ack_callback
        # Invoked when a LOCK/UNLOCK op-response (0xBB) is parsed, success or
        # failure, whichever side issued the command. In the captured traffic
        # op-responses answering our own commands were the normal case. The
        # only unsolicited op-responses observed were failure reports from
        # lock-started operations, and for one of those four frames the
        # absence of another connected central is only suggestive. An external
        # success settles through a status push and produces no op-response.
        # So this is the record of an operation ending, not a channel that
        # reports every operation started elsewhere.
        self._op_response_callback = op_response_callback
        # byte[15] of the most recent op-response: 0x00 success, non-zero =
        # OperationError enum value (MECH_* = jam). None until the first op.
        # Retained so a follow-up can expose the failure reason as a
        # diagnostic.
        self._last_op_error: int | None = None
        # The opcode of the operation this instance is awaiting an op-response
        # for, armed when the command write completes and cleared when the
        # operation ends. An op-response this does not match was
        # produced by an operation started at the lock, in the app, or by
        # auto-lock, or by an operation of ours whose wait already ended: the
        # wait clears this field on its way out, so a late op-response arrives
        # with nothing awaited. Nothing on our side reports its result.
        self._awaited_operation_opcode: int | None = None
        self._disconnected = False
        self._disconnect_callback = disconnect_callback
        self._disconnected_futures: set[asyncio.Future[None]] = set()

    def set_name(self, name: str) -> None:
        self.name = name

    def disconnected(self, *args: Any, **kwargs: Any) -> None:
        _LOGGER.debug("%s: Disconnected from lock callback", self.name)
        self._disconnected = True
        for future in self._disconnected_futures:
            if not future.done():
                future.set_result(None)
        self._disconnected_futures.clear()
        if self._disconnect_callback:
            self._disconnect_callback()

    async def connect(self, max_attempts: int = MAX_CONNECT_ATTEMPTS) -> None:
        """Connect to the lock."""
        _LOGGER.debug(
            "%s: Connecting to the lock",
            self.name,
        )
        try:
            self.client = await establish_connection(
                BleakClientWithServiceCache,
                self.ble_device_callback(),
                self.name,
                self.disconnected,
                use_services_cache=True,
                ble_device_callback=self.ble_device_callback,
                max_attempts=max_attempts,
            )
        except (TimeoutError, BleakError) as err:
            _LOGGER.error("%s: Failed to connect to the lock: %s", self.name, err)
            raise
        _LOGGER.debug("%s: Connected", self.name)

        self.session = Session(
            self.client,
            self.name,
            self._lock,
            self._disconnected_futures,
            self._internal_state_callback,
        )
        self.secure_session = SecureSession(
            self.client,
            self.name,
            self._lock,
            self._disconnected_futures,
            self.key_index,
        )
        session = self.session
        secure_session = self.secure_session
        _char_map = {
            session._read_characteristic: session.read_characteristic,
            session._write_characteristic: session.write_characteristic,
            secure_session._read_characteristic: secure_session.read_characteristic,
            secure_session._write_characteristic: secure_session.write_characteristic,
        }
        for char_uuid, char in _char_map.items():
            if not char:
                await self._handle_missing_characteristic(char_uuid)

        # Order matters here, we must start notify for the secure session before
        # the non-secure session
        await self.secure_session.start_notify()
        try:
            await self._setup_session()
        except BleakError as err:
            error_desc = str(err).lower()
            if "invalid handle" in error_desc:
                await self._handle_missing_characteristic(error_desc)
            raise
        await self.session.start_notify()

    async def _handle_missing_characteristic(self, char_uuid: str) -> None:
        """Handle missing characteristic."""
        self.secure_session = None
        self.session = None
        client = cast(BleakClientWithServiceCache, self.client)
        _LOGGER.warning(
            "%s: Missing characteristic %s; Clearing cache", self.name, char_uuid
        )
        await client.clear_cache()
        raise BleakError(f"Missing characteristic {char_uuid}")

    def _run_stream_hook(self, hook: Callable[[], None] | None, hook_name: str) -> None:
        """Run a notify-stream hook, containing anything it raises.

        The hooks run on the parse path, and Session._notify parses a frame
        before it resolves either wait future, so an exception escaping here
        would cost the frame its state update and abandon the staged wait of
        an operation whose motor may be running. A raising hook is a bug in
        the caller, which is why it is surfaced at error level.
        """
        if hook is None:
            return
        try:
            hook()
        except Exception:
            _LOGGER.exception(
                "%s: %s raised, continuing to parse the frame", self.name, hook_name
            )

    def _parse_state(self, state: bytes) -> Iterable[LockStateValue] | None:
        if state[0] == 0xBB:
            # Op-response for LOCK/UNLOCK (0xBB + 0x0A/0x0B), emitted when the
            # motor stops. The operation result is byte[15]: 0x00 = success,
            # any non-zero = failure (0x1E-0x23 = MECH_* motor stall / jam).
            if (
                state[1] in (Commands.LOCK.value, Commands.UNLOCK.value)
                and len(state) > 0x0F
            ):
                self._run_stream_hook(
                    self._op_response_callback, "op_response_callback"
                )
                result = state[0x0F]
                self._last_op_error = result
                if result != OperationError.COMM_SUCCESS:
                    solicited = state[1] == self._awaited_operation_opcode
                    if solicited and result in MECHANICAL_OPERATION_ERRORS:
                        # The known failure class of an operation we issued: the
                        # motor stalled, a jam. The caller is told by the
                        # OperationFailedError this op-response goes on to
                        # raise, which carries the same decoded cause, so this
                        # record is for the frame stream only.
                        _LOGGER.debug(
                            "%s: Operation failed with result 0x%02X (%s)",
                            self.name,
                            result,
                            _describe_operation_error(result),
                        )
                    else:
                        # Either the result is unexpected in a lock/unlock
                        # op-response, or no operation of ours is awaiting this
                        # opcode, which makes it an operation started at the
                        # lock, in the app, or by auto-lock, or one of ours
                        # whose wait already ended and cleared the awaited
                        # opcode before this late op-response landed. Nothing
                        # else reports these cases, so this is their only
                        # record.
                        _LOGGER.warning(
                            "%s: Operation failed with result 0x%02X (%s)",
                            self.name,
                            result,
                            _describe_operation_error(result),
                        )
                    # Every failure class needs manual intervention at the
                    # lock, so all of them display as JAMMED.
                    return [LockStatus.JAMMED]
                return ()  # success: recognized, no state update
            if state[1] == Commands.LOCK_ACTIVITY.value:
                return ()  # Ignore lock activity as these are historical events
            if state[1] == Commands.GETSTATUS.value:
                if state[4] == StatusType.LOCK_ONLY.value:
                    lock_status = state[0x08]
                    return [VALUE_TO_LOCK_STATUS.get(lock_status, LockStatus.UNKNOWN)]
                if state[4] == StatusType.DOOR_ONLY.value:
                    door_status = state[0x08]
                    return [VALUE_TO_DOOR_STATUS.get(door_status, DoorStatus.UNKNOWN)]
                if state[4] == StatusType.DOOR_AND_LOCK.value:
                    return self._parse_lock_and_door_state(state)
                if state[4] == StatusType.BATTERY.value:
                    return [self._parse_battery_state(state)]
            elif (
                state[1] == Commands.WRITESETTING.value
                or state[1] == Commands.READSETTING.value
            ):
                if state[4] == SettingType.AUTOLOCK.value:
                    return [self._parse_auto_lock_state(state)]
        elif state[0] == 0xAA:
            if state[1] in (Commands.UNLOCK.value, Commands.LOCK.value):
                # Operation acknowledgement: it echoes the request type and
                # carries no result nor resultant state
                self._run_stream_hook(self._ack_callback, "ack_callback")
                return ()
            if state[1] in (
                Commands.READSETTING.value,
                Commands.WRITESETTING.value,
            ):
                # ACK for a settings command (for example auto-lock, setting
                # 0x28). It carries no state -- the value arrives in the 0xBB
                # settings response -- so
                # recognize and ignore it rather than logging "Unknown state".
                # Kept specific to the settings opcodes so a new ACK type on
                # another model still surfaces as an unknown frame.
                return ()
        return None

    def _internal_state_callback(self, state: bytes) -> None:
        """Handle state change."""
        _LOGGER.debug("%s: State changed: %s", self.name, state.hex())
        parsed_state = self._parse_state(state)
        if parsed_state is None:
            # Unrecognized frame - surface it for diagnosis.
            _LOGGER.info("%s: Unknown state: %s", self.name, state.hex())
        elif parsed_state:
            # Non-empty iterable - emit the state(s) to the consumer.
            self._state_callback(parsed_state)
        # else: empty () - a recognized frame that carries no state update; ignore.

    async def _setup_session(self) -> None:
        """Setup the session."""
        assert self.session is not None  # nosec
        assert self.secure_session is not None  # nosec
        _LOGGER.debug("%s: Setting up the session", self.name)
        self.secure_session.set_key(self.key)
        handshake_keys = os.urandom(16)

        # Send SEC_LOCK_TO_MOBILE_KEY_EXCHANGE
        cmd = self.secure_session.build_command(0x01)
        util._copy(cmd, handshake_keys[0x00:0x08], destLocation=0x04)
        response = await self.secure_session.execute(
            cmd, "SEC_LOCK_TO_MOBILE_KEY_EXCHANGE"
        )
        if response[0x00] != 0x02:
            raise AuthError(
                "Authentication error: key or slot (key index) is incorrect: "
                "unexpected response to SEC_LOCK_TO_MOBILE_KEY_EXCHANGE: "
                + response.hex()
            )

        self.is_secure = True

        session_key = bytearray(16)
        util._copy(session_key, handshake_keys[0x00:0x08])
        util._copy(session_key, response[0x04:0x0C], destLocation=0x08)
        self.secure_session.set_key(session_key)

        # Send SEC_INITIALIZATION_COMMAND
        cmd = self.secure_session.build_command(0x03)
        util._copy(cmd, handshake_keys[0x08:0x10], destLocation=0x04)
        response = await self.secure_session.execute(cmd, "SEC_INITIALIZATION_COMMAND")
        if response[0] != 0x04:
            raise AuthError(
                "Authentication error: key or slot (key index) is incorrect: "
                "unexpected response to SEC_INITIALIZATION_COMMAND: " + response.hex()
            )
        self.session.set_key(session_key)
        self.secure_session.enable_cooldown()
        self.session.enable_cooldown()

    @raise_if_not_connected
    async def lock_info(self) -> LockInfo:
        """Probe the lock for information."""
        _LOGGER.debug("%s: Probing the lock", self.name)
        assert self.client is not None  # nosec
        # Read model first since it drives battery and door sense behavior.
        # Manufacturer name is not read because newer locks have a BLE
        # controller bug that corrupts ACL packets when reading that
        # characteristic, which kills the connection.
        char_uuids = (
            MODEL_NUMBER_CHARACTERISTIC,
            SERIAL_NUMBER_CHARACTERISTIC,
            FIRMWARE_REVISION_CHARACTERISTIC,
        )
        results: dict[str, str] = {}
        try:
            async with util.asyncio_timeout(LOCK_INFO_TIMEOUT):
                for char_uuid in char_uuids:
                    char = self.client.services.get_characteristic(char_uuid)
                    if not char:
                        _LOGGER.warning(
                            "%s: Characteristic %s not found", self.name, char_uuid
                        )
                        continue
                    try:
                        results[char_uuid] = (
                            (await self.client.read_gatt_char(char))
                            .decode()
                            .split("\0")[0]
                        )
                    except BleakError as err:
                        _LOGGER.warning(
                            "%s: Failed to read characteristic %s: %s",
                            self.name,
                            char_uuid,
                            err,
                        )
        except TimeoutError:
            _LOGGER.warning(
                "%s: Timeout reading lock info, using %d partial results",
                self.name,
                len(results),
            )
        # Use the BLE address as fallback serial to keep devices unique
        # in Home Assistant when the characteristic read fails.
        serial_fallback = self.ble_device_callback().address
        self._lock_info = LockInfo(
            manufacturer="Yale/August",
            model=results.get(MODEL_NUMBER_CHARACTERISTIC, ""),
            serial=results.get(SERIAL_NUMBER_CHARACTERISTIC, serial_fallback),
            firmware=results.get(FIRMWARE_REVISION_CHARACTERISTIC, "Unknown"),
        )
        return self._lock_info

    async def _execute_operation_command(
        self,
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        """Run a mechanical operation; returns only on a reported success.

        Completes only on the operation's own op-response (or an error): the
        staged session wait accepts the acknowledgement that echoes this
        command (opcode + operation byte) and then only the 0xBB op-response
        with the same opcode. The matcher values are captured from the command
        bytes here, BEFORE the session encrypts the buffer in place.

        write_success_callback belongs to the operation, not to this instance:
        it fires the moment the command's GATT write completes, which is the
        caller's single state-action moment, so only a caller that issued this
        command can reach it. The acknowledgement and op-response hooks are
        instance state by contrast: they observe the notify stream and must
        fire for operations we did not issue.

        Raises OperationFailedError when the op-response carries a non-zero
        result in byte[15]: the exchange completed and the lock named the
        failure (a MECH_* code is a motor stall, a jam).
        """
        assert self.session is not None  # nosec
        opcode = command[0x01]
        operation_byte = command[0x04]
        if progress is None:
            progress = OperationProgress()

        def _arm_awaited_opcode() -> None:
            # The written command is the first moment an op-response of ours
            # can exist. Arming at the call instead would cover the cooldown
            # wait, the session lock and the write itself, and an op-response
            # arriving in that stretch belongs to another central or to an
            # operation of ours that has already ended: no wait of ours is
            # armed to take it either.
            self._awaited_operation_opcode = opcode
            if write_success_callback is not None:
                write_success_callback()

        try:
            response = await self.session.execute_operation(
                command,
                command_name,
                ack_matcher=_ack_matcher(opcode, operation_byte),
                response_matcher=_operation_response_matcher(opcode),
                response_timeout=response_timeout,
                progress=progress,
                write_success_callback=_arm_awaited_opcode,
                wait_for_ack=wait_for_ack,
            )
        finally:
            # Cleared unconditionally, with no check that this operation is
            # the one that armed it: the field holds an opcode, not a token,
            # so two operations of the same kind could not be told apart by
            # one. It assumes a single operation at a time on an instance,
            # which PushLock's operation lock provides; a direct Lock consumer
            # running two at once would have the first to finish clear the
            # second's arming, and the only consequence is that the second's
            # own failure record logs at warning rather than debug.
            self._awaited_operation_opcode = None
        result = response[0x0F]
        if result != OperationError.COMM_SUCCESS:
            raise OperationFailedError(
                f"{self.name}: {command_name} reported failure 0x{result:02X} "
                f"({_describe_operation_error(result)})",
                result,
            )

    @raise_if_not_connected
    async def force_securemode(
        self, write_success_callback: Callable[[], None] | None = None
    ) -> None:
        """Force the lock into securemode."""
        _LOGGER.debug("%s: Securing", self.name)
        assert self.session is not None  # nosec
        await self._execute_operation_command(
            self.session.build_operation_command(
                Commands.LOCK, SECUREMODE_OPERATION_BYTE
            ),
            "force_securemode",
            response_timeout=OPERATION_RESPONSE_TIMEOUT,
            write_success_callback=write_success_callback,
        )
        _LOGGER.debug("%s: Finished securemode", self.name)

    @raise_if_not_connected
    async def force_lock(
        self, write_success_callback: Callable[[], None] | None = None
    ) -> None:
        """Force the lock to lock."""
        _LOGGER.debug("%s: Locking", self.name)
        assert self.session is not None  # nosec
        await self._execute_operation_command(
            self.session.build_command(Commands.LOCK),
            "force_lock",
            response_timeout=OPERATION_RESPONSE_TIMEOUT,
            write_success_callback=write_success_callback,
        )
        _LOGGER.debug("%s: Finished locking", self.name)

    @raise_if_not_connected
    async def force_unlock(
        self, write_success_callback: Callable[[], None] | None = None
    ) -> None:
        """Force the lock to unlock."""
        _LOGGER.debug("%s: Unlocking", self.name)
        assert self.session is not None  # nosec
        await self._execute_operation_command(
            self.session.build_command(Commands.UNLOCK),
            "force_unlock",
            response_timeout=OPERATION_RESPONSE_TIMEOUT,
            write_success_callback=write_success_callback,
        )
        _LOGGER.debug("%s: Finished unlocking", self.name)

    @raise_if_not_connected
    async def force_unlatch(
        self, write_success_callback: Callable[[], None] | None = None
    ) -> None:
        """Force the lock to unlatch (momentary "open door" / retract the latch).

        The Unlock opcode with operation byte 0x0A (there is no dedicated
        unlatch opcode); its op-response is a normal Unlock (0xBB 0A) frame.

        A repeated unlatch fires the latch again, opening the door again, so
        once the command write has been ATTEMPTED no failure may re-send it: a
        write call that errors can still have delivered the request (the PDU
        leaves the radio and only the ATT response is lost), so every failure
        from the write attempt onward converts to the non-retryable
        UnlatchError (OperationIncompleteError and OperationFailedError,
        already non-retryable, propagate as themselves). Failures before the
        write attempt (connect, session setup, encryption) stay retryable.

        That same rule makes the acknowledgement a don't care here, so this is
        the one operation that waits on its op-response alone.
        """
        _LOGGER.debug("%s: Unlatching", self.name)
        assert self.session is not None  # nosec
        progress = OperationProgress()
        try:
            await self._execute_operation_command(
                self.session.build_operation_command(
                    Commands.UNLOCK, UNLATCH_OPERATION_BYTE
                ),
                "force_unlatch",
                response_timeout=UNLATCH_OPERATION_RESPONSE_TIMEOUT,
                progress=progress,
                write_success_callback=write_success_callback,
                wait_for_ack=False,
            )
        except (OperationIncompleteError, OperationFailedError):
            # Already non-retryable: the result never arrived, or it arrived
            # and named a failure. Ordered ahead of the broad clause below so
            # both types reach the caller unwrapped.
            raise
        except Exception as err:
            # Broad on purpose, and every path here re-raises, so no failure
            # is discarded. The rule is stronger than any list of retryable
            # types: after a write attempt no failure of any kind may reach
            # the retry wrapper, because a retry re-sends the unlatch.
            # Converting to UnlatchError, which is not in the retry set,
            # enforces that even when the retry set widens. Cancellation is a
            # BaseException and passes through untouched. An AuthError raised
            # after the write is converted here too, so the auth failure
            # history does not advance and no failed AuthState is published
            # from this path. The reauth ladder advances on the next update
            # cycle instead.
            if progress.write_attempted:
                raise UnlatchError(
                    f"{self.name}: Unlatch failed after the command write was "
                    "attempted; not retried because the command may have "
                    "reached the lock and a repeated unlatch fires the latch "
                    "again"
                ) from err
            raise
        _LOGGER.debug("%s: Finished unlatching", self.name)

    @raise_if_not_connected
    async def set_auto_lock(self, mode: AutoLockMode, duration: int) -> None:
        """Change the auto lock setting (0x28).

        The value is two little-endian uint16 timers laid consecutively at
        [8:12]: the never-opened timer at [8:10] and the door-close timer at
        [10:12]. Both zero = off; ``n`` in the never-opened timer alone =
        instant with an ``n`` second delay; a timed relock sets both timers to
        the same ``n`` seconds. There is no mode byte -- the mode is implied by
        which timers are set, and the read side reports the never-opened timer
        (see ``_parse_auto_lock_state``).
        """
        _LOGGER.debug(
            "%s: Setting auto lock to mode=%d, dur=%d", self.name, mode, duration
        )
        assert self.session is not None  # nosec
        if mode == AutoLockMode.OFF or duration == 0:
            value = 0
        elif not 1 <= duration <= 0xFFFE:
            # Each timer is a uint16; cap conservatively below the field
            # maximum.
            raise ValueError(f"Auto lock duration out of range (1-65534): {duration}")
        elif mode == AutoLockMode.TIMER:
            value = duration | (duration << 16)
        else:  # INSTANT
            value = duration

        cmd = self.session.build_operation_command(
            Commands.WRITESETTING, SettingType.AUTOLOCK
        )
        util._copy(cmd, util._int_to_bytes(value, 4), destLocation=0x08)
        await self.session.execute(
            cmd,
            "set_auto_lock",
            _settings_response_matcher(
                Commands.WRITESETTING.value, SettingType.AUTOLOCK.value
            ),
        )
        _LOGGER.debug("%s: Finished setting auto lock", self.name)

    async def securemode(self) -> None:
        """Set securemode; raises OperationFailedError on a reported failure.

        The lock's own status is read first, and a lock already in securemode
        skips the operation.
        """
        if (await self.lock_status()) != LockStatus.SECUREMODE:
            await self.force_securemode()

    async def lock(self) -> None:
        """Lock; raises OperationFailedError on a reported failure.

        The lock's own status is read first, and a lock already locked skips
        the operation.
        """
        if (await self.lock_status()) != LockStatus.LOCKED:
            await self.force_lock()

    async def unlock(self) -> None:
        """Unlock; raises OperationFailedError on a reported failure.

        The lock's own status is read first, and a lock already unlocked
        skips the operation.
        """
        if (await self.lock_status()) != LockStatus.UNLOCKED:
            await self.force_unlock()

    async def unlatch(self) -> None:
        """Unlatch; raises OperationFailedError on a reported failure.

        Unlatch is a momentary "open door" action, not a steady state, so it
        always fires: lock() and unlock() short-circuit when the lock is
        already in the target state, and there is no "unlatched" steady state
        to test.
        """
        await self.force_unlatch()

    async def _execute_command(
        self, opcode: int, cmd_byte: int, command_name: str
    ) -> bytes:
        assert self.session is not None  # nosec
        command = self.session.build_operation_command(opcode, cmd_byte)
        _LOGGER.debug("%s: send: [%s] [%s]", self.name, command.hex(), hex(cmd_byte))
        response = await self.session.execute(command, command_name)
        _LOGGER.debug(
            "%s: response: [%s] [%s]", self.name, response.hex(), hex(cmd_byte)
        )
        return response

    def _parse_lock_and_door_state(
        self, response: bytes
    ) -> tuple[LockStatus, DoorStatus]:
        """Parse the lock and door state from the response."""
        return self._parse_lock_status(response[0x08]), self._parse_door_status(
            response[0x09]
        )

    def _parse_lock_status(self, lock_status: int) -> LockStatus:
        """Parse the lock state from the response."""
        lock_status_enum = VALUE_TO_LOCK_STATUS.get(lock_status, LockStatus.UNKNOWN)
        if lock_status_enum == LockStatus.UNKNOWN:
            _LOGGER.info(
                "%s: Unrecognized lock_status_str code: %s", self.name, hex(lock_status)
            )
        return lock_status_enum

    def _parse_door_status(self, door_status: int) -> DoorStatus:
        """Parse the door state from the response."""
        door_status_enum = VALUE_TO_DOOR_STATUS.get(door_status, DoorStatus.UNKNOWN)
        if door_status_enum == DoorStatus.UNKNOWN:
            _LOGGER.info(
                "%s: Unrecognized door_status_str code: %s", self.name, hex(door_status)
            )
        return door_status_enum

    def _parse_auto_lock_state(self, response: bytes) -> AutoLockState:
        """Parse the auto lock state from a READSETTING (setting 0x28) response.

        The auto-lock time is two little-endian uint16 timers laid
        consecutively at ``response[8:12]``; there is no mode byte. The
        never-opened timer is at [8:10] and the door-close timer at [10:12]. A
        Timed relock sets both timers to the same seconds, so a non-zero
        door-close timer signals Timed. A value with only the never-opened
        timer set is Instant; both zero is off. The mode is derived from which
        timers are set, never read from a wire byte.

        The Timed decode reports the never-opened timer: it is the field the
        app displays, and releases before the two-timer encoding stored the
        user's seconds there, so those values read back as set. A zero
        never-opened timer falls back to the door-close timer. A write derived
        from this state sets both timers to the same value.
        """
        value = util._bytes_to_int(response[0x08:0x0C])
        if value == 0:
            return AutoLockState(AutoLockMode.OFF, 0)
        never_opened = value & 0xFFFF
        door_close = (value >> 16) & 0xFFFF
        if door_close:
            return AutoLockState(AutoLockMode.TIMER, never_opened or door_close)
        return AutoLockState(AutoLockMode.INSTANT, never_opened)

    @raise_if_not_connected
    async def lock_status(self) -> LockStatus:
        _LOGGER.debug("%s: Executing lock_status", self.name)
        # We used to use 0x2F here but it seems to be broken on some locks
        response = await self._execute_command(
            Commands.GETSTATUS, StatusType.LOCK_ONLY, "lock_status"
        )
        _LOGGER.debug("%s: Finished executing lock_status", self.name)
        return self._parse_lock_status(response[0x08])

    @raise_if_not_connected
    async def door_status(self) -> DoorStatus:
        _LOGGER.debug("%s: Executing door_status", self.name)
        # We used to use 0x2F here but it seems to be broken on some locks
        response = await self._execute_command(
            Commands.GETSTATUS, StatusType.DOOR_ONLY, "door_status"
        )
        _LOGGER.debug("%s: Finished executing door_status", self.name)
        return self._parse_door_status(response[0x08])

    def _parse_battery_state(self, response: bytes) -> BatteryState:
        """Parse the battery state from the response."""
        voltage = util._bytes_to_int(response[0x08:0x0A]) / 1000
        # The voltage is divided by 4 in the lock
        # since it uses 4 AA batteries. For the Li-ion
        # battery, this is likely wrong, but since we don't
        # currently have a way to detect the battery type,
        # this is the best we can do for now.
        percentage = convert_voltage_to_percentage(voltage / 4)
        return BatteryState(voltage, percentage)

    @raise_if_not_connected
    async def battery(self) -> BatteryState:
        _LOGGER.debug("%s: Executing battery", self.name)
        response = await self._execute_command(
            Commands.GETSTATUS, StatusType.BATTERY, "battery"
        )
        _LOGGER.debug("%s: Finished executing battery", self.name)
        return self._parse_battery_state(response)

    @raise_if_not_connected
    async def auto_lock_status(self) -> None:
        """Request the auto-lock setting.

        The wait completes on the READSETTING acknowledgment, whose value field
        is zero, so it carries no setting; there is nothing to return. The
        stored setting arrives moments later as a 0xBB settings response on the
        notify path, which the push layer decodes and applies. Waiting for that
        settings response instead would stall the poll for the full write
        timeout on a lock that never answers READSETTING (no auto-lock support).
        """
        _LOGGER.debug("%s: Executing auto_lock_status", self.name)
        await self._execute_command(
            Commands.READSETTING,
            SettingType.AUTOLOCK,
            "auto_lock_status",
        )
        _LOGGER.debug("%s: Finished executing auto_lock_status", self.name)

    def _parse_unix_timestamp(self, timestamp_bytes: bytes) -> datetime:
        """Parse the unix timestamp to datetime from the bytes."""
        _LOGGER.debug(
            "%s: Parsing unix timestamp: %s", self.name, timestamp_bytes.hex()
        )
        unix_timestamp = int.from_bytes(timestamp_bytes, byteorder="little")
        _LOGGER.debug("%s: Parsed unix timestamp: %d", self.name, unix_timestamp)
        return datetime.fromtimestamp(unix_timestamp)

    def _parse_operation_source(
        self,
        source: int,
        remote_type: int,
    ) -> tuple[LockOperationSource, LockOperationRemoteType | None]:
        source_enum = VALUE_TO_LOCK_OPERATION_SOURCE.get(
            source, LockOperationSource.UNKNOWN
        )
        remote_type_enum: LockOperationRemoteType | None
        remote_type_enum = VALUE_TO_LOCK_OPERATION_REMOTE_TYPE.get(
            remote_type, LockOperationRemoteType.UNKNOWN
        )
        if source_enum is LockOperationSource.UNKNOWN:
            _LOGGER.info(
                "%s: Unrecognized operation source code: %s",
                self.name,
                hex(source),
            )

        if source_enum is LockOperationSource.REMOTE:
            if remote_type_enum is LockOperationRemoteType.UNKNOWN:
                _LOGGER.info(
                    "%s: Unrecognized operation remote type: %s",
                    self.name,
                    hex(remote_type),
                )
        else:
            remote_type_enum = None

        return source_enum, remote_type_enum

    def _parse_lock_activity(
        self, response: bytes
    ) -> DoorActivity | LockActivity | None:
        """Parse the lock activity from the response."""
        # We only know a subset of lock activities currently
        # response[0x04] seems to be the activity type
        # the rest of the response is data for the activity,
        # format seems to be specific to each individual activity type
        activity_type = response[0x04]
        _LOGGER.debug("%s: Activity type: 0x%02X", self.name, activity_type)
        if activity_type == LockActivityType.NONE.value:
            _LOGGER.debug("%s: No activity", self.name)
            return None

        if activity_type == LockActivityType.DOOR.value:
            # Timestamp is at 0x05-0x08
            # Door status is at 0x09
            timestamp = self._parse_unix_timestamp(response[0x05:0x09])
            door_status = self._parse_door_status(response[0x09])
            return DoorActivity(timestamp, door_status)
        if activity_type == LockActivityType.LOCK.value:
            # Timestamp is at 0x08-0x0B
            # Lock status is at 0x06
            # Operation source is at 0x05
            # Remote type is at 0x07
            timestamp = self._parse_unix_timestamp(response[0x08:0x0C])
            lock_status = self._parse_lock_status(response[0x06])
            operation_source, remote_type = self._parse_operation_source(
                response[0x05], response[0x07]
            )

            return LockActivity(
                timestamp,
                lock_status,
                source=operation_source,
                remote_type=remote_type,
            )
        if activity_type == LockActivityType.PIN.value:
            # Timestamp is at 0x05-0x08
            # Slot is at 0x0A
            # Lock status seems to be at lower half of 0x0C
            timestamp = self._parse_unix_timestamp(response[0x05:0x09])
            pin_slot = response[0x0A]
            lock_status = self._parse_lock_status(response[0x0C] & 0x0F)

            return LockActivity(
                timestamp,
                lock_status,
                source=LockOperationSource.PIN,
                slot=pin_slot,
            )
        _LOGGER.warning("%s: Unknown activity type: 0x%02X", self.name, activity_type)
        return None

    @raise_if_not_connected
    async def lock_activity(self) -> DoorActivity | LockActivity | None:
        _LOGGER.debug("%s: Executing lock_activity", self.name)
        assert self.session is not None  # nosec
        response = await self.session.execute(
            self.session.build_command(Commands.LOCK_ACTIVITY.value), "lock_activity"
        )
        _LOGGER.debug("%s: Finished executing lock_activity", self.name)
        return self._parse_lock_activity(response)

    async def disconnect(self) -> None:
        """Disconnect from the lock."""
        _LOGGER.debug("%s: Disconnecting from the lock", self.name)
        if not self.client or not self.client.is_connected:
            return

        try:
            await self._shutdown_connection()
        except BleakError:
            _LOGGER.debug(
                "%s: Failed to shutdown connection to lock", self.name, exc_info=True
            )
        finally:
            try:
                await self.client.disconnect()
            except BleakError:
                _LOGGER.debug(
                    "%s: Failed to disconnect from lock", self.name, exc_info=True
                )

    async def _shutdown_connection(self) -> None:
        """Shutdown the connection."""
        _LOGGER.debug("%s: Shutting down the connection", self.name)
        if self.session:
            await self.session.stop_notify()
        if not self.is_secure or not self.secure_session or self._disconnected:
            return
        cmd = self.secure_session.build_command(0x05)
        cmd[0x11] = 0x00
        response = None
        try:
            response = await self.secure_session.execute(cmd, "shutdown")
            await self.secure_session.stop_notify()
        except (AuthError, DisconnectedError):
            # Lock already disconnected us
            return
        except (TimeoutError, BleakError, EOFError) as err:
            if not util.is_disconnected_error(err):
                _LOGGER.debug(
                    "%s: Failed to cleanly disconnect from lock: %s", self.name, err
                )
            return
        except YaleXSBLEError as err:
            _LOGGER.debug(
                "%s: Failed to cleanly disconnect from lock: %s", self.name, err
            )
            return
        if response and response[0] != 0x8B:
            _LOGGER.debug(
                "%s: Unexpected response to DISCONNECT: %s", self.name, response.hex()
            )

    @property
    def is_connected(self) -> bool:
        """Return True if the lock is connected."""
        return bool(
            self.client
            and self.client.is_connected
            and self.session
            and self.secure_session
            and not self._disconnected
        )
