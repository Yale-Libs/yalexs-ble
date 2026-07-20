from __future__ import annotations

import asyncio
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
from .const import READ_CHARACTERISTIC, WRITE_CHARACTERISTIC

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

    async def _write(self, command: bytearray, command_name: str) -> bytes:
        """Write under the lock."""
        async with self._lock:
            return await self._locked_write(command, command_name)

    def _notify(self, char: int, data: bytearray) -> None:
        self._last_callback_time = time.monotonic()
        _LOGGER.debug(
            "%s: Receiving response via notify: %s (waiting=%s)",
            self.name,
            data.hex(),
            bool(self._notify_future),
        )
        decrypted_data = self.decrypt(data)
        if self._state_callback:
            self._state_callback(decrypted_data)
        _LOGGER.debug(
            "%s: Decrypted response via notify: %s", self.name, decrypted_data.hex()
        )
        notify_future = self._notify_future
        if notify_future is None:
            return
        if notify_future.done():
            # The awaiter was already cancelled by a timeout or disconnect;
            # clear the slot and ignore the notify. Logged above debug because
            # a lock that chronically answers late is a real fault that would
            # otherwise be invisible in installs without debug logging.
            _LOGGER.warning(
                "%s: Ignoring notify; the waiting request already gave up", self.name
            )
            self._notify_future = None
            return
        try:
            self._validate_response(data)
        except ResponseError as ex:
            _LOGGER.debug("%s: Invalid response, waiting for next one", self.name)
            notify_future.set_exception(ex)
            self._notify_future = None
            return
        notify_future.set_result(decrypted_data)
        self._notify_future = None

    async def _locked_write(self, command: bytearray, command_name: str) -> bytes:
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

        try:
            for attempt in range(3):
                future: asyncio.Future[bytes] = self.loop.create_future()
                self._notify_future = future
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
            # Drop the reference even on success — _notify already cleared it,
            # but on timeout / BleakError / cancellation it would point at a
            # cancelled or unresolved future, and a late notify would hit
            # set_result() on a done future (InvalidStateError).
            self._notify_future = None
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

    async def execute(self, command: bytearray, command_name: str) -> bytes:
        """Execute command."""
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
                return await self._write(command, command_name)
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
