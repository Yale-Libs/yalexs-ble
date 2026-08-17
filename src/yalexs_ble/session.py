from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from async_interrupt import interrupt
from bleak import BleakClient
from bleak_retry_connector import BleakError
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    CipherContext,
    algorithms,
    modes,
)

from . import util
from .const import READ_CHARACTERISTIC, WRITE_CHARACTERISTIC

_LOGGER = logging.getLogger(__name__)

COOLDOWN_TIME = 0.25

# Every frame the lock sends is 18 bytes: a 16-byte encrypted block plus two
# trailing plain bytes, and the checksum runs over all 18. The gate is strict
# equality, not a floor. A shorter payload must not reach the cipher: the
# cipher context consumes ciphertext in 16-byte blocks, so a partial block
# stays buffered inside it and every later frame on the connection decrypts
# misaligned, a state only a reconnect's set_key rebuilds. A longer payload
# would otherwise validate on its first 18 bytes and be passed on with a tail
# the cipher never saw.
RESPONSE_FRAME_LEN = 0x12

# Budget for the whole wait on the plain execute path. Sized above the ~6 s
# link supervision timeout so a dead link surfaces as a disconnect before this
# timer fires.
RESPONSE_TIMEOUT = 10

# Budget for stage 1 of a mechanical operation: the GATT write and the
# acknowledgement that follows it. The acknowledgement carries no mechanical
# delay, so this is a delivery budget, not a motion budget, and it is sized
# off the link rather than off the operation: the floor is the 6 s slow-mode
# supervision timeout, so a dead link presents as a disconnect rather than as
# an expired stage, and the margin above it covers the write and the time the
# stack takes to notice. Kept separate from RESPONSE_TIMEOUT so the plain path
# and this stage move independently.
ACK_TIMEOUT = 8.0

# Budget for a mechanical operation as a whole, measured from the command
# write to the op-response. It sits beside the stage it has to outlast: both
# deadlines run from the attempt start, so whatever the acknowledgement stage
# spends comes out of this one, and a value at or below ACK_TIMEOUT reaches
# stage 2 with nothing left. An operation whose motion is longer passes its
# own budget instead.
OPERATION_RESPONSE_TIMEOUT = 12.0

# Budget for an unlatch, the longest motion (retract the deadbolt, pull the
# spring latch in, hold it, release it), so it gets longer than a lock or an
# unlock.
UNLATCH_OPERATION_RESPONSE_TIMEOUT = 20.0


class YaleXSBLEError(Exception):
    """Base class for YaleXSBLE errors."""


class AuthError(YaleXSBLEError):
    """Error during authentication."""


class ResponseError(YaleXSBLEError):
    """Error during response."""


class DisconnectedError(YaleXSBLEError):
    """Disconnected during response."""


class NoAdvertisementError(YaleXSBLEError):
    """No advertisement data."""


class BluetoothError(YaleXSBLEError):
    """Bluetooth error."""


class OperationIncompleteError(YaleXSBLEError):
    """The operation's result never arrived.

    The command reached the lock (it was written, and normally acknowledged),
    so the motor may have run, but the op-response that reports the result was
    not received: timeout or disconnect mid-operation. Deliberately NOT a
    ResponseError and NOT in the bleak retry set: it must pass the retry
    decorator, because it ends the attempt ladder. Retries belong to the
    acknowledgement stage; past it the failure is reported up with the result
    unknown, for the caller to decide.
    """


class OperationFailedError(YaleXSBLEError):
    """The lock reported that an operation failed.

    The op-response arrived and its result byte named a failure, so the
    exchange completed and the outcome is known: the operation failed, not
    the link. A MECH_* code (0x1E to 0x23) is a motor stall, a jam; any
    other code names its own cause. Every failure class needs manual
    intervention at the lock, so all of them display as JAMMED.
    Deliberately NOT a ResponseError and NOT in the bleak retry set:
    re-driving a failed mechanism is not a recovery, so the failure passes
    the retry decorator to the caller unchanged.
    """

    def __init__(self, message: str, result: int) -> None:
        super().__init__(message)
        self.result = result

    def __reduce__(self) -> tuple[type[OperationFailedError], tuple[str, int]]:
        # BaseException rebuilds a copied or pickled exception by calling the
        # class with self.args, which carries the message alone, so the
        # default would call this __init__ an argument short. Both arguments
        # are named here instead. The message stays the only member of args,
        # so str() is unchanged.
        return (self.__class__, (str(self), self.result))


class UnlatchError(YaleXSBLEError):
    """An unlatch failed once its command write had been attempted.

    From the moment the write is attempted the command may have reached the
    lock, and a repeated unlatch fires the latch again, opening the door
    again, so no failure from that point on may re-send it. Deliberately NOT a
    ResponseError so it passes the retry decorator to the caller unchanged.
    """


@dataclass
class OperationProgress:
    """How far a mechanical command travelled, for stage-aware error policy.

    write_attempted is set immediately before the write call rather than
    after it, because a write call that errors may still have delivered the
    command: the request PDU can leave the radio with only the ATT response
    lost. From that mark onward the command may have reached the lock, which
    is what a caller that must never re-send one keys on.

    acknowledged and result are recorded where their frames arrive rather
    than where the staged wait resumes, because a disconnect can resolve in
    the same event-loop turn and cancel that wait before it ever resumes.
    """

    write_attempted: bool = False
    acknowledged: bool = False
    result: bytes | None = None


class Session:
    _write_characteristic = WRITE_CHARACTERISTIC
    _read_characteristic = READ_CHARACTERISTIC

    def __init__(
        self,
        client: BleakClient,
        name: str,
        lock: asyncio.Lock,
        disconnected_futures: set[asyncio.Future[None]],
        state_callback: Callable[[bytes], None] | None = None,
    ) -> None:
        """Init the session."""
        self.name = name
        self._lock = lock
        self.cipher_decrypt: CipherContext | None = None
        self.cipher_encrypt: CipherContext | None = None
        self.client = client
        self.write_characteristic = client.services.get_characteristic(
            self._write_characteristic
        )
        self.read_characteristic = client.services.get_characteristic(
            self._read_characteristic
        )
        self._notifications_started = False
        self._notify_future: asyncio.Future[bytes] | None = None
        # When set, only a frame this predicate accepts resolves the pending
        # future; other valid frames are passed to the state callback and the
        # wait continues. None = the first valid frame resolves (default).
        self._response_matcher: Callable[[bytes], bool] | None = None
        # Armed alongside the response future for mechanical operations: the
        # typed wait for the command's acknowledgement (0xAA echoing the
        # written opcode). Both futures are armed before the write so no frame
        # can fall into a gap between the two wait stages.
        self._ack_future: asyncio.Future[bytes] | None = None
        self._ack_matcher: Callable[[bytes], bool] | None = None
        # The progress record of the operation whose staged wait is in flight,
        # None between operations. Held for both stages, so it doubles as the
        # mark of a staged wait: the acknowledgement's own arming is cleared the
        # moment it arrives, and the op-response stage that follows looks
        # exactly like a plain wait.
        self._operation_progress: OperationProgress | None = None
        self._state_callback = state_callback
        self._disconnected_futures = disconnected_futures
        self._first_request = True
        self._last_callback_time = -86400.0
        self._enable_cooldown = False
        self.loop = asyncio.get_running_loop()

    def set_key(self, key: bytes | bytearray) -> None:
        self.cipher_encrypt = Cipher(
            algorithms.AES(key),
            modes.CBC(bytes(0x10)),  # nosec
        ).encryptor()
        self.cipher_decrypt = Cipher(
            algorithms.AES(key),
            modes.CBC(bytes(0x10)),  # nosec
        ).decryptor()

    def enable_cooldown(self) -> None:
        """Enable cooldown after each request."""
        self._enable_cooldown = True

    def decrypt(self, data: bytes | bytearray) -> bytes:
        if self.cipher_decrypt is not None:
            cipherText = data[0x00:0x10]
            plainText = self.cipher_decrypt.update(cipherText)
            if type(data) is not bytearray:
                data = bytearray(data)
            util._copy(data, plainText)

        return bytes(data)

    def build_operation_command(self, opcode: int, cmd_byte: int) -> bytearray:
        """Build a command to send to the lock."""
        cmd = self.build_command(opcode)
        cmd[0x04] = cmd_byte
        return cmd

    def build_command(self, opcode: int) -> bytearray:
        cmd = bytearray(0x12)
        cmd[0x00] = 0xEE
        cmd[0x01] = opcode
        cmd[0x10] = 0x02
        return cmd

    def _write_checksum(self, command: bytearray) -> None:
        checksum = util._simple_checksum(command)
        command[0x03] = checksum

    def _validate_response(self, response: bytes | bytearray) -> None:
        _LOGGER.debug(
            "%s: Response simple checksum: %s",
            self.name,
            str(util._simple_checksum(response)),
        )
        if util._simple_checksum(response) != 0:
            raise ResponseError(f"Simple checksum mismatch {response!r}")

        if response[0x00] != 0xBB and response[0x00] != 0xAA:
            raise ResponseError(f"Incorrect flag in response: {response[0x00]}")

    async def _write(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        """Write under the lock."""
        async with self._lock:
            return await self._locked_write(command, command_name, response_matcher)

    def _reject_frame(
        self,
        ex: ResponseError,
        frame: bytes | bytearray,
        level: int = logging.INFO,
    ) -> None:
        """Dispose of a frame that failed admission.

        The frame is withheld from the state callback. A live plain solicited
        wait receives the error, so _locked_write re-sends its command until
        its attempts run out and it raises. A wait that has already ended is
        left to its own cleanup, and an unsolicited frame has no waiter at
        all; either way the frame is dropped.
        """
        # The drop line carries the frame hex: these frames should not occur at
        # all, so a drop and its evidence are visible without turning debug on,
        # where the frames themselves are logged.
        _LOGGER.log(
            level, "%s: dropping invalid frame %s: %s", self.name, frame.hex(), ex
        )
        if self._operation_progress is not None:
            # A staged operation wait is in flight. Surfacing the error would
            # end the wait and re-send a mechanical command whose result is
            # unknown, so the frame is dropped and the wait continues; the stage
            # timeout is the backstop. Not expected in practice: the BLE link
            # layer's CRC and retransmission keep corrupted frames away from us.
            _LOGGER.debug(
                "%s: Invalid frame during an operation wait, still waiting", self.name
            )
            return
        if self._notify_future is None:
            return
        if self._notify_future.done():
            # The wait ended before this frame arrived: its timeout cancelled
            # the future, and the waiting task has not resumed yet to disarm
            # the slot. Resolving a future that is already done raises
            # InvalidStateError into the notify dispatcher, so the slot is left
            # to the waiter's own cleanup.
            return
        _LOGGER.debug("%s: Invalid response, waiting for next one", self.name)
        self._notify_future.set_exception(ex)
        self._notify_future = None
        self._response_matcher = None

    def _notify(self, char: int, data: bytearray) -> None:
        self._last_callback_time = time.monotonic()
        _LOGGER.debug(
            "%s: Receiving response via notify: %s (waiting=%s)",
            self.name,
            data.hex(),
            self._notify_future is not None and not self._notify_future.done(),
        )
        if len(data) != RESPONSE_FRAME_LEN:
            # This gate must sit ahead of decrypt (see RESPONSE_FRAME_LEN):
            # dropping the payload here keeps the cipher aligned, so the
            # session stays usable for the next good frame.
            self._reject_frame(
                ResponseError(
                    f"{len(data)}-byte payload is not an "
                    f"{RESPONSE_FRAME_LEN}-byte response frame"
                ),
                data,
                # A short payload is a corrupt frame off the air, which the
                # radio produces on its own. An over-length one is not: the
                # lock builds nothing longer, so it says the transport below
                # is doing something nobody has seen, and it is worth a level
                # more attention.
                logging.WARNING if len(data) > RESPONSE_FRAME_LEN else logging.INFO,
            )
            return
        decrypted_data = self.decrypt(data)
        _LOGGER.debug(
            "%s: Decrypted response via notify: %s", self.name, decrypted_data.hex()
        )
        try:
            # Runs on every frame, not only while a wait is armed: the state
            # callback below drives the consumer's state, so an unsolicited
            # frame has to clear the same bar as a solicited one.
            self._validate_response(decrypted_data)
        except ResponseError as ex:
            self._reject_frame(ex, decrypted_data)
            return
        # Every frame is handed to the state path, the one that answers a
        # solicited wait included, and it is handed over before the wait is
        # resolved below. A caller that reads the lock discards what the read
        # returns and relies on the frame having been applied here, so a change
        # that moves this hand-off behind an await, or drops it for the frame
        # that answers, silently stops every polled reading reaching the
        # display.
        if self._state_callback:
            self._state_callback(decrypted_data)
        if (
            (progress := self._operation_progress) is not None
            and self._ack_future is not None
            and self._ack_matcher is not None
            and self._ack_matcher(decrypted_data)
        ):
            self._ack_future.set_result(decrypted_data)
            self._ack_future = None
            self._ack_matcher = None
            # Recorded where the acknowledgement is observed rather than where
            # the staged wait resumes: a disconnect can resolve in the same
            # event-loop turn and cancel that wait before it ever resumes, and
            # an acknowledgement recorded nowhere is classified as a drop
            # before the acknowledgement, which is retryable and re-sends a
            # command the lock has already taken.
            progress.acknowledged = True
            return
        if self._notify_future is None:
            return
        if self._response_matcher is not None and not self._response_matcher(
            decrypted_data
        ):
            # A valid frame that does not complete the pending command (a
            # foreign push, a status settle, an unexpected ack): the state
            # callback has already seen it; the wait continues.
            _LOGGER.debug(
                "%s: Frame does not complete the pending command, still waiting",
                self.name,
            )
            return
        if progress is not None:
            # The op-response, recorded here for the same reason the
            # acknowledgement is: a disconnect resolving in this turn cancels
            # the wait before it resumes, and a result recorded nowhere is
            # reported as an operation whose result never arrived, when in
            # fact it did.
            progress.result = decrypted_data
        if self._notify_future.done():
            # The same window as above, reached by a frame that would otherwise
            # answer the command. The record above is taken first, so a wait
            # that ended in this turn still reports the result it was waiting
            # for. The waiter reports its own timeout, and that report is all a
            # reader of the log would otherwise see, so the frame that would
            # have answered it is named here.
            _LOGGER.debug(
                "%s: dropping the answer to a wait that already ended: %s",
                self.name,
                decrypted_data.hex(),
            )
            return
        self._notify_future.set_result(decrypted_data)
        self._notify_future = None
        self._response_matcher = None

    def _encrypt_command(self, command: bytearray, command_name: str) -> None:
        # NOTE: The last two bytes are not encrypted
        # General idea seems to be that if the last byte
        # of the command indicates an offline key offset (is non-zero),
        # the command is "secure" and encrypted with the offline key
        assert self.cipher_encrypt is not None, "Cipher not set"  # nosec
        plainText = command[0x00:0x10]
        cipherText = self.cipher_encrypt.update(plainText)
        util._copy(command, cipherText)
        _LOGGER.debug(
            "%s: Encrypted command %s: %s", self.name, command_name, command.hex()
        )

    async def _locked_write(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        if not self.client.is_connected:
            raise BleakError("disconnected")
        self._encrypt_command(command, command_name)

        for attempt in range(3):
            future: asyncio.Future[bytes] = self.loop.create_future()
            self._notify_future = future
            self._response_matcher = response_matcher
            _LOGGER.debug(
                "%s: Writing command to %s: %s",
                self.name,
                self.write_characteristic,
                command.hex(),
            )
            _LOGGER.debug("%s: Waiting for response", self.name)
            try:
                async with util.asyncio_timeout(RESPONSE_TIMEOUT):
                    try:
                        await self.client.write_gatt_char(
                            self.write_characteristic, command, True
                        )
                        result = await future
                    except ResponseError:
                        # Only reachable outside a staged operation wait:
                        # _notify skips a corrupt frame while _operation_progress
                        # is set, so a mechanical command is never re-written
                        # from here.
                        if attempt == 2:
                            raise
                        _LOGGER.debug("%s: Invalid response, retrying", self.name)
                        continue
                    else:
                        break
            finally:
                # If THIS attempt's future is still armed we are exiting
                # abnormally (timeout / cancellation / disconnect / a write
                # that raised): disarm it, so a frame arriving before the next
                # command re-arms the slot does not land on a future nobody is
                # waiting on. The next command re-arms both slots before its
                # own write, so what leaks here is only this window.
                if self._notify_future is future:
                    self._notify_future = None
                    self._response_matcher = None
        _LOGGER.debug("%s: Got response: %s", self.name, result.hex())
        return result

    async def _locked_write_operation(
        self,
        command: bytearray,
        command_name: str,
        ack_matcher: Callable[[bytes], bool],
        response_matcher: Callable[[bytes], bool],
        response_timeout: float,
        progress: OperationProgress,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> bytes:
        """Write a mechanical command and run the staged wait.

        Starting (or restarting) an operation initialises its timers: both
        stage deadlines run from the attempt start. Stage 1 (ACK_TIMEOUT)
        covers the GATT write and the typed acknowledgement; stage 2 (the
        operation's response_timeout, also from attempt start) covers the
        op-response, the physical end of movement. Only an op-response, or an
        error, ends the wait.

        With wait_for_ack False the acknowledgement stage is skipped and the
        op-response is awaited for the whole budget. The GATT write keeps its
        own ACK_TIMEOUT bound either way: that one bounds the local write
        exchange, which carries no mechanical delay.
        """
        if not self.client.is_connected:
            raise BleakError("disconnected")
        self._encrypt_command(command, command_name)

        attempt_start = time.monotonic()
        ack_future: asyncio.Future[bytes] = self.loop.create_future()
        result_future: asyncio.Future[bytes] = self.loop.create_future()
        # Arm both stages before the write: the op-response can follow the
        # acknowledgement within milliseconds on a fast link, so re-arming
        # between stages would race it.
        self._ack_future = ack_future
        self._ack_matcher = ack_matcher
        self._notify_future = result_future
        self._response_matcher = response_matcher
        self._operation_progress = progress
        try:
            _LOGGER.debug(
                "%s: Writing command to %s: %s",
                self.name,
                self.write_characteristic,
                command.hex(),
            )
            # A write call that errors may still have delivered the command:
            # the request PDU can leave the radio with only the ATT response
            # lost, so from this point an error leaves delivery unknown.
            progress.write_attempted = True
            async with util.asyncio_timeout(ACK_TIMEOUT):
                await self.client.write_gatt_char(
                    self.write_characteristic, command, True
                )
            if write_success_callback is not None:
                # The hook runs at the point of no return: the command is
                # delivered and the motor may already be moving. An exception
                # escaping here would abandon the staged wait and, being
                # retryable upstream, re-send a mechanical command, so the
                # exception is contained and the wait continues. A raising
                # hook is a bug in the caller, which is why it is surfaced at
                # error level.
                try:
                    write_success_callback()
                except Exception:
                    _LOGGER.exception(
                        "%s: write success callback for %s raised, "
                        "continuing the staged wait",
                        self.name,
                        command_name,
                    )
            if wait_for_ack:
                _LOGGER.debug("%s: Waiting for acknowledgement", self.name)
                ack_remaining = ACK_TIMEOUT - (time.monotonic() - attempt_start)
                done, _ = await asyncio.wait(
                    (ack_future, result_future),
                    timeout=max(ack_remaining, 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if result_future in done:
                    # The op-response supersedes the acknowledgement stage: it
                    # either beat the acknowledgement entirely, or both
                    # resolved in the same event-loop turn. Either way the
                    # op-response is the answer, and whether an acknowledgement
                    # also arrived was already recorded where it was received.
                    return result_future.result()
                if ack_future not in done:
                    raise TimeoutError(
                        f"{self.name}: No acknowledgement to {command_name} "
                        f"within {ACK_TIMEOUT}s of the command being issued"
                    )
            _LOGGER.debug("%s: Waiting for the op-response", self.name)
            result_remaining = response_timeout - (time.monotonic() - attempt_start)
            try:
                async with util.asyncio_timeout(max(result_remaining, 0)):
                    result = await result_future
            except TimeoutError as err:
                if (recorded := progress.result) is not None:
                    # The op-response arrived and was recorded where the frame
                    # landed; the stage-2 timer fired in the same event-loop
                    # turn and canceled the wait before it could resume with
                    # the result the future already holds. The result is
                    # known, so report it rather than the timeout.
                    _LOGGER.debug(
                        "%s: op-response to %s arrived in the same turn as "
                        "the stage-2 timeout; returning the recorded result",
                        self.name,
                        command_name,
                    )
                    return recorded
                raise OperationIncompleteError(
                    f"{self.name}: no op-response to {command_name} arrived "
                    f"within {response_timeout}s of the command being issued "
                    f"(acknowledged: {progress.acknowledged})"
                ) from err
        finally:
            # Unconditional: the session lock serialises operations, so the
            # record armed above is still this operation's own.
            self._operation_progress = None
            if self._ack_future is ack_future:
                self._ack_future = None
                self._ack_matcher = None
            if self._notify_future is result_future:
                self._notify_future = None
                self._response_matcher = None
        _LOGGER.debug("%s: Got op-response: %s", self.name, result.hex())
        return result

    async def start_notify(self) -> None:
        """Start notify."""
        if not self._notifications_started:
            _LOGGER.debug("%s: Starting notify for %s", self.name, type(self))
            try:
                await self._start_notify(self._notify)
            except BleakError as err:
                _LOGGER.debug("%s: Failed to start notify: %s", self.name, err)
                if "not found" in str(err):
                    raise AuthError(f"{self.name}: {err}") from err
                raise
            self._notifications_started = True

    async def _start_notify(self, callback: Callable[[int, bytearray], None]) -> None:
        """Start notify."""
        if not self.client.is_connected:
            return
        try:
            await self.client.start_notify(self.read_characteristic, callback)
            # Workaround for MacOS to allow restarting notify
        except ValueError:
            await self.stop_notify()
            if not self.client.is_connected:
                return
            await self.client.start_notify(self.read_characteristic, callback)

    async def stop_notify(self) -> None:
        """Stop notify."""
        if not self.client.is_connected or not self._notifications_started:
            return
        _LOGGER.debug("%s: Stopping notify: %s", self.name, type(self))
        try:
            await self.client.stop_notify(self.read_characteristic)
        except EOFError as err:
            _LOGGER.debug("%s: D-Bus stopping notify: %s", self.name, err)
        except BleakError as err:
            _LOGGER.debug("%s: Bleak error stopping notify: %s", self.name, err)

    async def _wait_for_cooldown(self) -> None:
        while (
            self._enable_cooldown
            and (cooldown_remain := time.monotonic() - self._last_callback_time)
            < COOLDOWN_TIME
        ):
            _LOGGER.debug(
                "%s: Waiting %s for lock to settle", self.name, cooldown_remain
            )
            # If we send commands to fast the lock may crash and stop
            # advertising. This is a workaround to avoid that since
            # it means a battery pull is required to recover.
            await asyncio.sleep(COOLDOWN_TIME - cooldown_remain)

    async def execute(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        """Execute command.

        ``response_matcher`` narrows which notify frame answers the command:
        valid frames that do not match still reach the state callback, but the
        solicited wait stays armed until a matching frame arrives, or until
        the wait ends some other way: this commit disarms it on a timeout, a
        cancellation, a disconnect and a write that raised alike. Without a
        matcher the first valid frame answers, as before.
        """
        await self._wait_for_cooldown()
        assert self.cipher_encrypt is not None, "Cipher not set"  # nosec
        self._write_checksum(command)
        disconnected_future = asyncio.get_running_loop().create_future()
        disconnected_futures = self._disconnected_futures
        disconnected_futures.add(disconnected_future)
        try:
            async with interrupt(
                disconnected_future, DisconnectedError, f"{self.name}: Disconnected"
            ):
                return await self._write(command, command_name, response_matcher)
        except BleakError as err:
            if self._first_request and util.is_key_error(err):
                raise AuthError(
                    f"Authentication error: key or slot (key index) is incorrect: {err}"
                ) from err
            if util.is_disconnected_error(err):
                raise DisconnectedError(f"{self.name}: {err}") from err
            raise
        finally:
            disconnected_futures.discard(disconnected_future)
            self._first_request = False

    async def execute_operation(
        self,
        command: bytearray,
        command_name: str,
        ack_matcher: Callable[[bytes], bool],
        response_matcher: Callable[[bytes], bool],
        response_timeout: float,
        progress: OperationProgress,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> bytes:
        """Execute a mechanical operation command with the staged wait.

        Error policy by stage (the caller's retry decorator sees the types).
        The acknowledgement carries no mechanical delay, so its absence
        signals a delivery problem and predicts a missing op-response: write
        and acknowledgement failures keep their retryable types (TimeoutError
        / DisconnectedError / BleakError) and the retry re-sends the command
        early rather than waiting out the op-response budget. Once the
        operation is acknowledged, a timeout or disconnect raises the
        non-retryable OperationIncompleteError, ending the attempt ladder: the
        result is unknown and the caller decides.

        response_timeout is the budget for the whole exchange, measured from
        the moment the command is issued. What is left for the op-response is
        that budget minus the time already spent, so it must leave room beyond
        the acknowledgement the operation waits for.
        OPERATION_RESPONSE_TIMEOUT is that budget for a lock or an unlock, and
        an operation with a longer motion passes its own.

        wait_for_ack=False drops the acknowledgement stage and waits only for
        the op-response, for the whole budget. The early delivery signal is
        worth having only where the caller may re-send, so an operation that
        forbids a re-send once the command is written gains nothing from it
        and would pay for it with the op-response: an acknowledgement lost on
        a link that is otherwise working would end the operation before the
        result it is waiting for could arrive. The acknowledgement is still
        matched and recorded when it lands; only the wait on it is dropped.
        """
        await self._wait_for_cooldown()
        assert self.cipher_encrypt is not None, "Cipher not set"  # nosec
        self._write_checksum(command)
        disconnected_future = asyncio.get_running_loop().create_future()
        disconnected_futures = self._disconnected_futures
        disconnected_futures.add(disconnected_future)
        try:
            async with (
                interrupt(
                    disconnected_future,
                    DisconnectedError,
                    f"{self.name}: Disconnected",
                ),
                self._lock,
            ):
                return await self._locked_write_operation(
                    command,
                    command_name,
                    ack_matcher,
                    response_matcher,
                    response_timeout,
                    progress,
                    write_success_callback,
                    wait_for_ack,
                )
        except DisconnectedError as err:
            if (result := progress.result) is not None:
                # The op-response arrived and was recorded where the frame
                # landed; the disconnect resolved in the same event-loop turn
                # and cancelled the wait before it could resume. The result is
                # known, so report it rather than the interruption.
                _LOGGER.debug(
                    "%s: Disconnected in the same turn as the op-response to "
                    "%s; returning the recorded result",
                    self.name,
                    command_name,
                )
                return result
            if progress.acknowledged:
                raise OperationIncompleteError(
                    f"{self.name}: Disconnected while awaiting the op-response "
                    f"to {command_name}; the result is unknown"
                ) from err
            raise
        except BleakError as err:
            if self._first_request and util.is_key_error(err):
                raise AuthError(
                    f"Authentication error: key or slot (key index) is incorrect: {err}"
                ) from err
            if util.is_disconnected_error(err):
                if (result := progress.result) is not None:
                    # As above: the op-response is in hand, only the wait was
                    # lost.
                    _LOGGER.debug(
                        "%s: Disconnected in the same turn as the op-response "
                        "to %s; returning the recorded result",
                        self.name,
                        command_name,
                    )
                    return result
                if progress.acknowledged:
                    raise OperationIncompleteError(
                        f"{self.name}: Disconnected while awaiting the "
                        f"op-response to {command_name}; the result is unknown"
                    ) from err
                raise DisconnectedError(f"{self.name}: {err}") from err
            raise
        finally:
            disconnected_futures.discard(disconnected_future)
            self._first_request = False
