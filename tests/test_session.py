"""Tests for the Session class — focused on _notify_future lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from bleak.exc import BleakError

from yalexs_ble import util as util_mod
from yalexs_ble.session import ResponseError, Session


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


def _valid_response() -> bytearray:
    """A checksum-valid response packet.

    Byte 0 is the response flag (0xBB/0xAA), byte 3 closes the running sum
    to 0.
    """
    response = bytearray(0x12)
    response[0x00] = 0xBB
    response[0x03] = (-sum(response[:0x12])) & 0xFF
    return response


@pytest.mark.asyncio
async def test_notify_future_cleared_after_successful_write() -> None:
    """On the happy path, _notify_future is None when _locked_write returns."""
    session = _build_session()
    command = bytearray(0x12)

    async def fake_write(*_args, **_kwargs):
        # Deliver the notify from a later loop iteration, matching bleak's
        # real ordering: the write returns, _locked_write suspends on
        # `await future`, and only then does the callback fire.
        session.loop.call_soon(session._notify, 0, _valid_response())

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
async def test_notify_future_cleared_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the response never arrives, the cancelled future ref is cleared."""
    session = _build_session()
    command = bytearray(0x12)

    # write_gatt_char succeeds but no notify ever arrives — the inner
    # asyncio_timeout(10) is too long for a unit test, so patch it down.
    # Bind the real callable first to avoid recursing into the patched one.
    real_timeout = util_mod.asyncio_timeout
    monkeypatch.setattr(util_mod, "asyncio_timeout", lambda _s: real_timeout(0.01))

    with pytest.raises(TimeoutError):
        await session._locked_write(command, "test-cmd")

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_invalid_response_retries_then_succeeds() -> None:
    """An invalid response fails the future, and the retry loop recovers."""
    session = _build_session()
    command = bytearray(0x12)
    responses = [bytearray(0x12), _valid_response()]  # first has a bad flag byte

    async def fake_write(*_args, **_kwargs):
        session.loop.call_soon(session._notify, 0, responses.pop(0))

    session.client.write_gatt_char.side_effect = fake_write

    await session._locked_write(command, "test-cmd")

    assert not responses
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_invalid_response_raises_after_final_attempt() -> None:
    """Three invalid responses exhaust the retries and propagate."""
    session = _build_session()
    command = bytearray(0x12)

    async def fake_write(*_args, **_kwargs):
        # All-zero packet: the checksum is valid, the flag byte is not.
        session.loop.call_soon(session._notify, 0, bytearray(0x12))

    session.client.write_gatt_char.side_effect = fake_write

    with pytest.raises(ResponseError):
        await session._locked_write(command, "test-cmd")

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_notify_without_pending_future_is_ignored() -> None:
    """A notify with no awaiter at all is a no-op."""
    session = _build_session()

    session._notify(0, _valid_response())

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_late_notify_after_timeout_is_ignored() -> None:
    """A notify arriving after timeout must not raise InvalidStateError."""
    session = _build_session()

    # Simulate the post-timeout state: _notify_future holds a cancelled future.
    cancelled = session.loop.create_future()
    cancelled.cancel()
    session._notify_future = cancelled

    # Should be a no-op — no exception propagates from the notify callback.
    session._notify(0, _valid_response())

    # And the stale reference is cleared so future calls start fresh.
    assert session._notify_future is None
