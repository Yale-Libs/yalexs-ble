"""A fake ESPHome Bluetooth proxy that a Lock connects through in the tests."""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from bleak.exc import BleakError

from yalexs_ble.const import READ_CHARACTERISTIC, SECURE_WRITE_CHARACTERISTIC
from yalexs_ble.lock import Lock
from yalexs_ble.util import _simple_checksum


class EspProxy:
    """A fake ESPHome Bluetooth proxy for one address."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.clients: list[EspClient] = []
        self.next_secure_write_error: BleakError | None = None
        self._waiting: list[asyncio.Future[None]] = []

    async def establish(
        self, _cls: Any, _device: Any, _name: str, disconnected: Any, **_: Any
    ) -> "EspClient":
        """Return a new connected client, in place of establish_connection."""
        self.log.append("connect")
        client = EspClient(self, disconnected)
        if (error := self.next_secure_write_error) is not None:
            self.next_secure_write_error = None
            client.next_write_error[SECURE_WRITE_CHARACTERISTIC] = error
        return client

    def request_disconnect(self) -> asyncio.Future[None]:
        self.log.append("disconnect request")
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiting.append(waiter)
        loop.call_later(0.005, self.send_connected_false, "connected=False")
        return waiter

    def send_connected_false(self, entry: str) -> None:
        self.log.append(entry)
        for waiter in self._waiting:
            if not waiter.done():
                waiter.set_result(None)
        self._waiting.clear()
        for client in list(self.clients):
            client.link_down()


class EspClient:
    """A client of EspProxy, whose is_connected turns False at the proxy's answer."""

    def __init__(self, proxy: EspProxy, disconnected: Any) -> None:
        self.proxy = proxy
        self.is_connected = True
        self.down = asyncio.Event()
        self.next_write_error: dict[str, BleakError] = {}
        self.hold_after_answer: asyncio.Event | None = None
        self._disconnected = disconnected
        self._notify: dict[str, Callable[[int, bytearray], None]] = {}
        self._writes: set[asyncio.Future[None]] = set()
        self.services = MagicMock()
        self.services.get_characteristic = lambda uuid: SimpleNamespace(uuid=uuid)
        proxy.clients.append(self)

    async def start_notify(self, char: Any, callback: Any) -> None:
        self._notify[char.uuid] = callback

    async def stop_notify(self, _char: Any) -> None:
        return None

    async def set_connection_params(self, *_: Any) -> None:
        return None

    async def clear_cache(self) -> bool:
        return True

    async def write_gatt_char(self, char: Any, data: bytes, _response: bool) -> None:
        if not self.is_connected:
            raise BleakError("not connected")
        if (error := self.next_write_error.pop(char.uuid, None)) is not None:
            await asyncio.sleep(0)
            raise error
        if char.uuid == SECURE_WRITE_CHARACTERISTIC:
            # Every secure write is answered as the shutdown command is, by the
            # lock ending the link.
            asyncio.get_running_loop().call_soon(
                self.proxy.send_connected_false, "the lock ends the link"
            )
            ended = asyncio.get_running_loop().create_future()
            self._writes.add(ended)
            await ended
        reply = bytearray(data)
        reply[0x00] = 0xBB
        reply[0x03] = 0
        reply[0x03] = _simple_checksum(reply)
        asyncio.get_running_loop().call_soon(
            self._notify[READ_CHARACTERISTIC], 0, reply
        )

    async def disconnect(self) -> None:
        answered = self.proxy.request_disconnect()
        try:
            await answered
        finally:
            self.link_down()
        if self.hold_after_answer is not None:
            await self.hold_after_answer.wait()

    def link_down(self) -> None:
        if not self.is_connected:
            return
        self.is_connected = False
        self.down.set()
        self.proxy.clients.remove(self)
        for ended in self._writes:
            if not ended.done():
                ended.set_exception(BleakError("changed connection status"))
        self._disconnected(self)


def esp_connections(proxy: EspProxy) -> AbstractContextManager[Any]:
    """Connect every Lock through proxy."""
    return patch("yalexs_ble.lock.establish_connection", proxy.establish)


@contextmanager
def esp_backend(proxy: EspProxy) -> Iterator[None]:
    """Connect every Lock through proxy, with identity ciphers."""

    async def setup_session(lock: Lock) -> None:
        for session in (lock.session, lock.secure_session):
            assert session is not None
            session.cipher_encrypt = MagicMock(update=bytes)
            session.cipher_decrypt = MagicMock(update=bytes)
        lock.is_secure = True

    with (
        esp_connections(proxy),
        patch.object(Lock, "_setup_session", setup_session),
    ):
        yield
