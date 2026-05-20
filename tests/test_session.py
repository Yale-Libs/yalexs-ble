"""Tests for the Session class — focused on _notify_future lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from bleak.exc import BleakError

from yalexs_ble import util as util_mod
from yalexs_ble.session import Session


def _build_session() -> Session:
    """Construct a Session with a stubbed BleakClient."""
    client = MagicMock()
    client.is_connected = True
    client.services.get_characteristic.return_value = MagicMock()
    client.write_gatt_char = AsyncMock()
    session = Session(
        client=client,
        name="lock",
        lock=asyncio.Lock(),
        disconnected_futures=set(),
    )
    # set_key normally happens during the handshake; bypass with a fake cipher.
    fake_cipher = MagicMock()
    fake_cipher.update.side_effect = lambda data: data
    session.cipher_encrypt = fake_cipher
    session.cipher_decrypt = fake_cipher
    return session


@pytest.mark.asyncio
async def test_notify_future_cleared_after_successful_write() -> None:
    """On the happy path, _notify_future is None when _locked_write returns."""
    session = _build_session()
    command = bytearray(0x12)

    async def fake_write(*_args, **_kwargs):
        # Simulate a notification arriving from the lock — byte 0 is the
        # response flag (0xBB/0xAA), byte 3 must close the running sum to 0.
        response = bytearray(0x12)
        response[0x00] = 0xBB
        response[0x03] = (-sum(response[:0x12])) & 0xFF
        session._notify(0, bytes(response))

    session.client.write_gatt_char.side_effect = fake_write

    await session._locked_write(command, "test-cmd")
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_notify_future_cleared_after_bleak_error() -> None:
    """If write_gatt_char raises, the orphaned future reference is cleared."""
    session = _build_session()
    command = bytearray(0x12)
    session.client.write_gatt_char.side_effect = BleakError("write failed")

    with pytest.raises(BleakError):
        await session._locked_write(command, "test-cmd")

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_notify_future_cleared_after_timeout() -> None:
    """If the response never arrives, the cancelled future ref is cleared."""
    session = _build_session()
    command = bytearray(0x12)

    # write_gatt_char succeeds but no notify ever arrives — the inner
    # asyncio_timeout(10) is too long for a unit test, so patch it down.
    real_timeout = util_mod.asyncio_timeout

    def short_timeout(_seconds):
        return real_timeout(0.01)

    util_mod.asyncio_timeout = short_timeout
    try:
        with pytest.raises(TimeoutError):
            await session._locked_write(command, "test-cmd")
    finally:
        util_mod.asyncio_timeout = real_timeout

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_late_notify_after_timeout_is_ignored() -> None:
    """A notify arriving after timeout must not raise InvalidStateError."""
    session = _build_session()

    # Simulate the post-timeout state: _notify_future holds a cancelled future.
    cancelled = session.loop.create_future()
    cancelled.cancel()
    session._notify_future = cancelled

    # A valid-looking response. _notify must not call set_result on it.
    response = bytearray(0x12)
    response[0x00] = 0xBB
    response[0x03] = (-sum(response[:0x12])) & 0xFF

    # Should be a no-op — no exception propagates from the notify callback.
    session._notify(0, bytes(response))

    # And the stale reference is cleared so future calls start fresh.
    assert session._notify_future is None
