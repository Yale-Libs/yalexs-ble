from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import monotonic

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
from .const import (
    READ_CHARACTERISTIC,
    RESPONSE_FRAME_LEN,
    WRITE_CHARACTERISTIC,
    OperationError,
)

_LOGGER = logging.getLogger(__name__)

COOLDOWN_TIME = 0.25

# How long execute() waits for the frame answering a command.
RESPONSE_TIMEOUT = 10.0

# Stage 1 of a mechanical operation: the GATT write plus its acknowledgment.
# Above the ~6 s link supervision timeout so a dead link reads as a disconnect.
ACK_TIMEOUT = 8.0

# Whole-operation budget, command write to op-response.
OPERATION_RESPONSE_TIMEOUT = 12.0

# The unlatch's op-response budget; the dwell after the latch pull is not
# part of it.
UNLATCH_OPERATION_RESPONSE_TIMEOUT = 12.0


class YaleXSBLEError(Exception):
    """Base class for YaleXSBLE errors."""


class AuthError(YaleXSBLEError):
    """Error during authentication."""


class ResponseError(YaleXSBLEError):
    """Error during response."""


class KeycodeError(YaleXSBLEError):
    """The lock reported a non-zero result for a keycode command."""

    def __init__(self, command: str, error: OperationError | int) -> None:
        self.command = command
        self.error = error
        name = error.name if isinstance(error, OperationError) else "unknown error"
        super().__init__(f"{command} failed: {name} (0x{error:02X})")


class DisconnectedError(YaleXSBLEError):
    """Disconnected during response."""


class NoAdvertisementError(YaleXSBLEError):
    """No advertisement data."""


class BluetoothError(YaleXSBLEError):
    """Bluetooth error."""


class OperationIncompleteError(YaleXSBLEError):
    """The operation ended without its op-response and was not retried."""


class OperationFailedError(YaleXSBLEError):
    """The lock reported that an operation failed; not retryable.

    Re-driving a failed mechanism is not a recovery.
    """

    def __init__(self, message: str, result: int) -> None:
        super().__init__(message)
        self.result = result

    def __reduce__(self) -> tuple[type[OperationFailedError], tuple[str, int]]:
        # self.args holds only the message, so the default rebuild would fail.
        return (self.__class__, (str(self), self.result))


class UnlatchError(YaleXSBLEError):
    """An unlatch failed once its write was attempted; not retryable.

    The command may have reached the lock, and a repeated unlatch opens the
    door again.
    """


@dataclass
class OperationProgress:
    """How far a mechanical operation got.

    Recorded on frame arrival and never reset, so pass a fresh instance per
    operation.
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
        # When set, only a matching frame resolves the pending future; other
        # valid frames still reach the state callback and the wait continues.
        self._notify_matcher: Callable[[bytes], bool] | None = None
        # Acknowledgment wait of a staged operation; armed with the response
        # future before the write.
        self._ack_future: asyncio.Future[bytes] | None = None
        self._ack_matcher: Callable[[bytes], bool] | None = None
        # Set for the whole staged wait; tells _notify a staged wait is active.
        self._operation_progress: OperationProgress | None = None
        # Called by _notify with the op-response of the staged wait.
        self._result_callback: Callable[[bytes], None] | None = None
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
        cmd = bytearray(RESPONSE_FRAME_LEN)
        cmd[0x00] = 0xEE
        cmd[0x01] = opcode
        cmd[0x10] = 0x02
        return cmd

    def _write_checksum(self, command: bytearray) -> None:
        command[0x03] = 0
        checksum = util._simple_checksum(command)
        command[0x03] = checksum

    def _validate_response(self, response: bytes | bytearray) -> None:
        checksum = util._simple_checksum(response)
        _LOGGER.debug("%s: Response simple checksum: %s", self.name, checksum)
        if checksum != 0:
            # The frame hex rides on the error, not only on the drop line:
            # when a command exhausts its attempts this error surfaces at
            # levels where the INFO drop line was never emitted.
            raise ResponseError(
                f"Simple checksum mismatch (expected 0, got {checksum}) "
                f"in frame {response.hex()}"
            )

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

    def _disarm_wait(self) -> asyncio.Future[bytes] | None:
        """Disarm the solicited wait and return its future, if one was armed.

        A caller that resolves the returned future must check done() first: a
        timeout or a disconnect cancels the future from the waiting side, and
        a frame that raced that cancellation must not resolve it again.
        """
        future = self._notify_future
        self._notify_future = None
        self._notify_matcher = None
        return future

    def _disarm_ack(self) -> asyncio.Future[bytes] | None:
        """Disarm the acknowledgment wait and return its future, if armed."""
        future = self._ack_future
        self._ack_future = None
        self._ack_matcher = None
        return future

    def _reject_frame(
        self,
        ex: ResponseError,
        frame: bytes | bytearray,
        level: int = logging.INFO,
    ) -> None:
        """Dispose of a frame that failed admission.

        The frame is withheld from the state callback.
        """
        # The drop line carries the frame hex: these frames should not occur at
        # all, so a drop and its evidence are visible without turning debug on,
        # where the frames themselves are logged.
        _LOGGER.log(
            level, "%s: dropping invalid frame %s: %s", self.name, frame.hex(), ex
        )
        if self._operation_progress is not None:
            # A staged wait never fails on a bad frame; its deadline is the backstop.
            _LOGGER.debug(
                "%s: Invalid frame during an operation wait, still waiting", self.name
            )
            return
        if (future := self._disarm_wait()) is not None and not future.done():
            future.set_exception(ex)

    def _notify(self, char: int, data: bytearray) -> None:
        self._last_callback_time = monotonic()
        _LOGGER.debug(
            "%s: Receiving response via notify: %s (waiting=%s)",
            self.name,
            data.hex(),
            bool(self._notify_future),
        )
        if not data:
            # An empty notification is a transport artifact, not a frame off
            # the lock: the stack emits them on its own, so one carries no
            # signal about the link or the command in flight, and it was a
            # no-op here before the length gate existed. A truncated frame is
            # different: it is evidence the response itself was corrupted, so
            # it is rejected below, while this is dropped without touching the
            # wait.
            _LOGGER.debug("%s: Dropping empty notification", self.name)
            return
        if len(data) != RESPONSE_FRAME_LEN:
            # Strict equality, and it must sit ahead of decrypt: the cipher
            # context consumes ciphertext in 16-byte blocks, so a partial
            # block fed to it stays buffered inside and desynchronizes every
            # later frame on the connection, a state only a reconnect's
            # set_key rebuilds. An over-length payload would validate on its
            # first 18 bytes and be passed on with a tail the cipher never
            # saw. (Dropping a truncation of genuine ciphertext still skips a
            # block the lock chained, so the next frame decrypts garbled and
            # is rejected too; the chain recovers on the frame after, where a
            # poisoned context never does.)
            self._reject_frame(
                ResponseError(
                    f"{len(data)}-byte payload is not an "
                    f"{RESPONSE_FRAME_LEN}-byte response frame"
                ),
                data,
                # An over-length frame is worth more attention than a short
                # one: the radio truncates frames on its own, but nothing on
                # the link builds a longer one, so it points at the transport
                # below.
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
        # Every frame that validates reaches _state_callback, the one answering
        # a read included, and it reaches it before the waiter below is resolved.
        # Callers that discard what a read returns rely on that order, so moving
        # this call after an await, or skipping it for the frame that answers a
        # read, resumes a caller before its answer is applied. SecureSession
        # inherits this method and is built without a state callback, which is
        # why the call is guarded.
        if self._state_callback:
            self._state_callback(decrypted_data)
        if (
            (progress := self._operation_progress) is not None
            and (ack_future := self._ack_future) is not None
            and self._ack_matcher is not None
            and self._ack_matcher(decrypted_data)
        ):
            self._disarm_ack()
            ack_future.set_result(decrypted_data)
            # Recorded on arrival; a disconnect may cancel the wait first.
            progress.acknowledged = True
            return
        if self._notify_future is None:
            return
        if self._notify_matcher is not None and not self._notify_matcher(
            decrypted_data
        ):
            # A valid frame, but not the answer this command is waiting for
            # (for example the 0xAA acknowledgment that precedes a settings response).
            # It has already been passed to the state callback above; keep the
            # future armed for the real answer.
            _LOGGER.debug(
                "%s: Response is not the awaited frame, waiting for next one",
                self.name,
            )
            return
        if progress is not None:
            progress.result = decrypted_data
            if (result_callback := self._result_callback) is not None:
                # Called on the frame, before the waiting task resumes, so a
                # frame handled in between cannot reach the state callback
                # ahead of the op-response reaching the caller. Contained so
                # the op-response still completes the wait.
                try:
                    result_callback(decrypted_data)
                except Exception:
                    _LOGGER.exception(
                        "%s: result callback raised, completing the staged wait",
                        self.name,
                    )
        if (future := self._disarm_wait()) is not None and not future.done():
            future.set_result(decrypted_data)

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

        future: asyncio.Future[bytes] | None = None
        try:
            # The loop never exhausts: the last attempt re-raises, so the only
            # ways out are break and raise.
            for attempt in range(3):  # pragma: no branch
                future = self.loop.create_future()
                self._notify_future = future
                self._notify_matcher = response_matcher
                _LOGGER.debug(
                    "%s: Writing command to %s: %s",
                    self.name,
                    self.write_characteristic,
                    command.hex(),
                )
                _LOGGER.debug("%s: Waiting for response", self.name)
                async with util.asyncio_timeout(RESPONSE_TIMEOUT):
                    try:
                        await self.client.write_gatt_char(
                            self.write_characteristic, command, True
                        )
                        result = await future
                    except ResponseError:
                        if attempt == 2:
                            raise
                        _LOGGER.debug("%s: Invalid response, retrying", self.name)
                        continue
                    else:
                        break
        finally:
            # A timeout or a disconnect interrupt leaves the wait armed with a
            # future the waiter has abandoned. Disarm it so a late frame
            # cannot leak this command's matcher into a later wait. (On the
            # paths that resolved the wait this is a no-op.)
            self._disarm_wait()
            # A frame can fail the wait while the GATT write itself is still
            # in flight; if the write then raises, the ResponseError set on
            # the future is never awaited. Retrieve it so asyncio does not
            # log "exception was never retrieved" with no context. suppress
            # covers the pending and cancelled states, where there is
            # nothing to retrieve. The None check guards only create_future
            # itself raising on the first attempt, which would otherwise turn
            # into a NameError here that masks the real error; no test can
            # reach it.
            if future is not None:  # pragma: no branch
                with contextlib.suppress(
                    asyncio.CancelledError, asyncio.InvalidStateError
                ):
                    future.exception()
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
        result_callback: Callable[[bytes], None] | None = None,
        wait_for_ack: bool = True,
    ) -> bytes:
        """Write a mechanical command, then wait for its acknowledgment and
        op-response, both timed from the write.

        ACK_TIMEOUT bounds the write and the acknowledgment, response_timeout
        the op-response. With wait_for_ack False the acknowledgment wait is
        skipped; the write keeps its ACK_TIMEOUT bound either way.
        """
        if not self.client.is_connected:
            raise BleakError("disconnected")
        self._encrypt_command(command, command_name)

        attempt_start = monotonic()
        ack_future: asyncio.Future[bytes] = self.loop.create_future()
        result_future: asyncio.Future[bytes] = self.loop.create_future()
        # Both armed before the write so no frame falls between the stages.
        self._ack_future = ack_future
        self._ack_matcher = ack_matcher
        self._notify_future = result_future
        self._notify_matcher = response_matcher
        self._operation_progress = progress
        self._result_callback = result_callback
        try:
            _LOGGER.debug(
                "%s: Writing command to %s: %s",
                self.name,
                self.write_characteristic,
                command.hex(),
            )
            # Set before the call: an errored write may still have delivered.
            progress.write_attempted = True
            async with util.asyncio_timeout(ACK_TIMEOUT):
                await self.client.write_gatt_char(
                    self.write_characteristic, command, True
                )
            if write_success_callback is not None:
                # Contained: the lock may already be running the operation.
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
                _LOGGER.debug("%s: Waiting for acknowledgment", self.name)
                ack_remaining = ACK_TIMEOUT - (monotonic() - attempt_start)
                done, _ = await asyncio.wait(
                    (ack_future, result_future),
                    timeout=max(ack_remaining, 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if result_future in done:
                    # The op-response supersedes the acknowledgment; a stale
                    # same-opcode op-response can land here (_completed logs it).
                    return self._completed(
                        result_future.result(), progress, command_name
                    )
                if ack_future not in done:
                    raise TimeoutError(
                        f"{self.name}: No acknowledgment to {command_name} "
                        f"within {ACK_TIMEOUT}s of the command being issued"
                    )
            _LOGGER.debug("%s: Waiting for the op-response", self.name)
            result_remaining = response_timeout - (monotonic() - attempt_start)
            # asyncio.wait never cancels the future, so a same-turn arrival
            # still reads as done.
            await asyncio.wait((result_future,), timeout=max(result_remaining, 0))
            if not result_future.done():
                never_acknowledged = (
                    ""
                    if progress.acknowledged
                    else "; the command was never acknowledged"
                )
                raise OperationIncompleteError(
                    f"{self.name}: No op-response to {command_name} arrived "
                    f"within {response_timeout}s of the command being issued"
                    f"{never_acknowledged}"
                )
            result = result_future.result()
        finally:
            # The session lock serializes operations, so these are this one's.
            self._operation_progress = None
            self._result_callback = None
            self._disarm_ack()
            self._disarm_wait()
        return self._completed(result, progress, command_name)

    def _completed(
        self, result: bytes, progress: OperationProgress, command_name: str
    ) -> bytes:
        """Return the op-response, logging when no acknowledgment preceded it.

        Every completion path returns through here; the log line is the only
        field trace of a dropped acknowledgment or a stale op-response.
        """
        if not progress.acknowledged:
            _LOGGER.info(
                "%s: %s completed on its op-response; no acknowledgment was received",
                self.name,
                command_name,
            )
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
            and (cooldown_remain := monotonic() - self._last_callback_time)
            < COOLDOWN_TIME
        ):
            _LOGGER.debug(
                "%s: Waiting %s for lock to settle", self.name, cooldown_remain
            )
            # If we send commands to fast the lock may crash and stop
            # advertising. This is a workaround to avoid that since
            # it means a battery pull is required to recover.
            await asyncio.sleep(COOLDOWN_TIME - cooldown_remain)

    def _raise_for_bleak_error(self, err: BleakError) -> None:
        """Raise AuthError or DisconnectedError for a BleakError that means one."""
        if self._first_request and util.is_key_error(err):
            raise AuthError(
                f"Authentication error: key or slot (key index) is incorrect: {err}"
            ) from err
        if util.is_disconnected_error(err):
            raise DisconnectedError(f"{self.name}: {err}") from err

    @contextlib.asynccontextmanager
    async def _command_scope(self, command: bytearray) -> AsyncIterator[None]:
        """Prepare a command and guard its exchange against a disconnect."""
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
                yield
        finally:
            disconnected_futures.discard(disconnected_future)

    async def execute(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        """Execute command.

        ``response_matcher`` narrows which notify frame answers the command:
        valid frames that do not match still reach the state callback, but the
        solicited wait stays armed until a matching frame arrives (or the
        write times out). Without a matcher the first valid frame answers, as
        before.
        """
        try:
            async with self._command_scope(command):
                return await self._write(command, command_name, response_matcher)
        except BleakError as err:
            self._raise_for_bleak_error(err)
            raise
        finally:
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
        result_callback: Callable[[bytes], None] | None = None,
        wait_for_ack: bool = True,
    ) -> bytes:
        """Run a mechanical operation with the staged wait and return its op-response.

        An op-response recorded before a failure is returned; a failure after the
        acknowledgment, or an op-response wait that runs to response_timeout,
        raises OperationIncompleteError; any other failure is raised as execute()
        would, for the caller to retry. response_timeout must exceed ACK_TIMEOUT.
        result_callback runs with the op-response as that frame is handled,
        before the wait resolves.
        """
        if progress.write_attempted or progress.acknowledged or progress.result:
            # A reused record would report a previous attempt's frames as this one's.
            raise ValueError(
                f"{self.name}: {command_name} needs a fresh OperationProgress"
            )
        try:
            async with self._command_scope(command), self._lock:
                return await self._locked_write_operation(
                    command,
                    command_name,
                    ack_matcher,
                    response_matcher,
                    response_timeout,
                    progress,
                    write_success_callback,
                    result_callback,
                    wait_for_ack,
                )
        except OperationIncompleteError:
            raise
        # Broad: the retry set is wider than BleakError.
        except Exception as err:
            if (result := progress.result) is not None:
                _LOGGER.debug(
                    "%s: %s failed after its op-response was recorded: %r; "
                    "returning the recorded result",
                    self.name,
                    command_name,
                    err,
                )
                return self._completed(result, progress, command_name)
            if progress.acknowledged:
                raise OperationIncompleteError(
                    f"{self.name}: {command_name} failed after the lock "
                    f"acknowledged it: {err!r}; the result is unknown"
                ) from err
            if isinstance(err, BleakError):
                self._raise_for_bleak_error(err)
            raise
        finally:
            self._first_request = False
