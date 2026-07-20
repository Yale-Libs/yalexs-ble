"""Tests for the low-level BLE Session request/response plumbing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from bleak import BleakError

from yalexs_ble.session import ResponseError, Session

KEY = bytes(range(16))


def _valid_response() -> bytearray:
    """Build an 18-byte response whose simple checksum validates."""
    data = bytearray(0x12)
    data[0x00] = 0xBB
    data[0x03] = (-0xBB) & 0xFF
    return data


def _invalid_response() -> bytearray:
    """Build a response that fails the simple checksum."""
    data = bytearray(0x12)
    data[0x00] = 0xBB
    return data


def _make_session() -> Session:
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    session = Session(client, "Test Lock", asyncio.Lock(), set())
    session.set_key(KEY)
    # Decrypt in place would scramble the crafted responses below; the encrypt
    # side is kept so _locked_write follows its real path.
    session.cipher_decrypt = None
    return session


@pytest.mark.asyncio
async def test_notify_ignores_late_response_for_finished_future() -> None:
    """A notify arriving after the awaiter gave up is dropped, not set."""
    session = _make_session()
    future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
    future.cancel()
    session._notify_future = future

    # Must not raise InvalidStateError.
    session._notify(0, _valid_response())

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_notify_without_waiter_is_a_no_op() -> None:
    """A notify with nothing waiting is ignored."""
    session = _make_session()
    session._notify(0, _valid_response())
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_locked_write_returns_decrypted_response() -> None:
    """A well-formed response resolves the pending write."""
    session = _make_session()

    async def _respond(*args: object, **kwargs: object) -> None:
        session._notify(0, _valid_response())

    session.client.write_gatt_char = AsyncMock(side_effect=_respond)

    result = await session._locked_write(bytearray(0x12), "test")

    assert len(result) == 0x12
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_locked_write_retries_on_invalid_response() -> None:
    """An invalid checksum retries the write rather than failing outright."""
    session = _make_session()
    attempts = 0

    async def _respond(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        session._notify(0, _invalid_response() if attempts == 1 else _valid_response())

    session.client.write_gatt_char = AsyncMock(side_effect=_respond)

    await session._locked_write(bytearray(0x12), "test")

    assert attempts == 2
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_locked_write_raises_after_three_invalid_responses() -> None:
    """Three invalid responses give up with a ResponseError."""
    session = _make_session()
    attempts = 0

    async def _respond(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        session._notify(0, _invalid_response())

    session.client.write_gatt_char = AsyncMock(side_effect=_respond)

    with pytest.raises(ResponseError):
        await session._locked_write(bytearray(0x12), "test")

    assert attempts == 3
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_locked_write_clears_notify_future_on_write_error() -> None:
    """A failed write leaves no dangling future for a late notify to hit."""
    session = _make_session()
    session.client.write_gatt_char = AsyncMock(side_effect=BleakError("boom"))

    with pytest.raises(BleakError):
        await session._locked_write(bytearray(0x12), "test")

    assert session._notify_future is None


@pytest.mark.asyncio
async def test_locked_write_raises_when_disconnected() -> None:
    """Writing on a disconnected client fails fast."""
    session = _make_session()
    session.client.is_connected = False

    with pytest.raises(BleakError):
        await session._locked_write(bytearray(0x12), "test")
