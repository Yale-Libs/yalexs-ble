from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

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
from .const import READ_CHARACTERISTIC, RESPONSE_FRAME_LEN, WRITE_CHARACTERISTIC

_LOGGER = logging.getLogger(__name__)

COOLDOWN_TIME = 0.25


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
        self._notify_matcher: Callable[[bytes], bool] | None = None
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

    def _reject_frame(
        self,
        ex: ResponseError,
        frame: bytes | bytearray,
        level: int = logging.INFO,
    ) -> None:
        """Dispose of a frame that failed admission.

        The frame is withheld from the state callback. A solicited wait still
        receives the error, so _locked_write re-sends its command until its
        attempts run out and it raises; an unsolicited frame has no waiter and
        is dropped.
        """
        # The drop line carries the frame hex: these frames should not occur at
        # all, so a drop and its evidence are visible without turning debug on,
        # where the frames themselves are logged.
        _LOGGER.log(
            level, "%s: dropping invalid frame %s: %s", self.name, frame.hex(), ex
        )
        if (future := self._disarm_wait()) is not None and not future.done():
            future.set_exception(ex)

    def _notify(self, char: int, data: bytearray) -> None:
        self._last_callback_time = time.monotonic()
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
            # different — it is evidence the response itself was corrupted —
            # so it fails the armed wait below and triggers a re-send, while
            # this is dropped without touching the wait.
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
        if (future := self._disarm_wait()) is not None and not future.done():
            future.set_result(decrypted_data)

    async def _locked_write(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        # NOTE: The last two bytes are not encrypted
        # General idea seems to be that if the last byte
        # of the command indicates an offline key offset (is non-zero),
        # the command is "secure" and encrypted with the offline key
        if not self.client.is_connected:
            raise BleakError("disconnected")
        assert self.cipher_encrypt is not None, "Cipher not set"  # nosec
        plainText = command[0x00:0x10]
        cipherText = self.cipher_encrypt.update(plainText)
        util._copy(command, cipherText)
        _LOGGER.debug(
            "%s: Encrypted command %s: %s", self.name, command_name, command.hex()
        )

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
                async with util.asyncio_timeout(10):
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
