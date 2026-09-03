"""Session-level tests for frame admission and the solicited-response matcher."""

import asyncio
import logging
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak import BleakError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from yalexs_ble import util
from yalexs_ble.const import Commands, SettingType
from yalexs_ble.lock import _settings_response_matcher
from yalexs_ble.secure_session import SecureSession
from yalexs_ble.session import RESPONSE_FRAME_LEN, ResponseError, Session

# Verbatim field frames (2026-07-16 capture, YUR/DEL fw 2.1.0): the READSETTING
# acknowledgment and, ~40 ms later, the 0xBB frame carrying the stored value
# (Timed 1800: both uint16 timers set to 1800).
READ_ACK = bytes.fromhex("aa0400282800000000000000000000000200")
READ_ANSWER = bytes.fromhex("bb0400fb2800000008070807000000000000")
# The same value frame with its checksum byte cleared, so _validate_response
# rejects it the way a corrupted frame off the air is rejected.
CORRUPT_ANSWER = bytes.fromhex("bb0400002800000008070807000000000000")


def _make_session(received: list[bytes], key: bytes | None = None) -> Session:
    """Create a Session with a mock client.

    Pass-through crypto by default; with ``key``, the real CBC contexts.
    """
    client = MagicMock(is_connected=True)
    session = Session(client, "testlock", asyncio.Lock(), set(), received.append)
    if key is None:
        session.decrypt = bytes  # type: ignore[method-assign, assignment]
        session.cipher_encrypt = MagicMock(update=bytes)
    else:
        session.set_key(key)
    return session


@pytest.mark.asyncio
async def test_settings_wait_skips_ack_and_completes_on_value_frame() -> None:
    """With a matcher, the acknowledgment does not answer the command.

    The 0xAA acknowledgment arrives first and must leave the wait armed (its
    zero value field would decode as auto-lock off); the 0xBB settings frame
    resolves it. Both frames still reach the state callback.
    """
    received: list[bytes] = []
    session = _make_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )

    async def deliver(*_args: object, **_kwargs: object) -> None:
        # The wait is armed before the GATT write, so both response frames
        # can be delivered from here, in their on-air order.
        session._notify(0, bytearray(READ_ACK))
        # The acknowledgment must not have resolved the wait.
        assert session._notify_future is not None
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status", matcher)

    assert result == READ_ANSWER
    assert received == [READ_ACK, READ_ANSWER]


@pytest.mark.asyncio
async def test_execute_without_matcher_takes_first_valid_frame() -> None:
    """Without a matcher the first valid frame answers, as before."""
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ACK))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status")

    assert result == READ_ACK


@pytest.mark.asyncio
async def test_the_answering_frame_reaches_the_callback_before_the_wait() -> None:
    """The frame answering a command is applied before the wait it resolves.

    A caller that reads the lock discards what the read returns and relies on
    the frame having been applied here, so the answering frame has to reach
    _state_callback, and has to reach it while the wait is still armed. Skip
    the call for that frame and the callback never sees it; move it below the
    resolution and the caller can resume ahead of it.
    """
    received: list[bytes] = []
    armed_at_hand_off: list[bool] = []
    session = _make_session(received)

    def state_callback(data: bytes) -> None:
        received.append(data)
        armed_at_hand_off.append(session._notify_future is not None)

    session._state_callback = state_callback

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ACK))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status")

    assert result == READ_ACK
    assert received == [READ_ACK]
    assert armed_at_hand_off == [True]


@pytest.mark.asyncio
async def test_corrupt_frame_disarms_the_wait_and_the_command_is_retried() -> None:
    """A frame that fails the checksum ends the wait, and the write is repeated.

    The corrupt frame resolves the future with a ResponseError, so the matcher
    must be cleared with it -- otherwise the retry re-arms the wait with the
    previous command's matcher still in place.
    """
    received: list[bytes] = []
    session = _make_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )
    frames = [CORRUPT_ANSWER, READ_ANSWER]

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(frames.pop(0)))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status", matcher)

    assert result == READ_ANSWER
    assert session.client.write_gatt_char.await_count == 2
    # The corrupt frame was withheld from the state callback; only the valid
    # one reached it, and only it answered.
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_corrupt_frame_on_every_attempt_raises_and_leaves_the_slot_empty() -> (
    None
):
    """Three corrupt frames exhaust the retries and the error reaches the caller."""
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(CORRUPT_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    with pytest.raises(ResponseError):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session.client.write_gatt_char.await_count == 3
    assert session._notify_future is None
    assert session._notify_matcher is None


def _short_timeout(_seconds: float) -> object:
    """Replacement for util.asyncio_timeout that expires almost immediately."""
    return asyncio.timeout(0.01)


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_success() -> None:
    """On the happy path the notify slot is empty when _locked_write returns."""
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session._locked_write(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_timeout() -> None:
    """When no response arrives, the timeout path disarms the notify slot."""
    received: list[bytes] = []
    session = _make_session(received)
    # The write succeeds but no notify is ever delivered.
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_late_notify_after_timeout_is_ignored() -> None:
    """A frame arriving after the timeout cleared the slot is a no-op.

    The timeout leaves ``_notify_future`` at None, so a late frame reaches the
    state callback but resolves no wait and raises nothing.
    """
    received: list[bytes] = []
    session = _make_session(received)
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    session._notify(0, bytearray(READ_ANSWER))
    assert session._notify_future is None
    # The late frame still reached the state callback.
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_fresh_command_rearms_slot_after_timeout() -> None:
    """After a timeout, a fresh command re-arms the slot and completes normally."""
    received: list[bytes] = []
    session = _make_session(received)
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")
    assert session._notify_future is None

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session._locked_write(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session._notify_future is None


@pytest.mark.parametrize("checksum", [util._simple_checksum, util._security_checksum])
def test_the_checksum_helpers_refuse_a_short_buffer(
    checksum: Callable[[bytes], int],
) -> None:
    """Both checksum helpers defend their own input.

    A slice over a short buffer would silently checksum whatever bytes are
    there — potentially summing to a "valid" result — so the helpers raise
    instead of leaning on the notify gate being their only caller.
    """
    with pytest.raises(ValueError, match="checksum needs 18 bytes, got 17"):
        checksum(b"\x00" * 17)


@pytest.mark.asyncio
async def test_a_frame_with_an_unknown_flag_is_withheld_from_state() -> None:
    """A valid checksum alone does not admit a frame with an unknown flag.

    The flag check used to run only while a wait was armed; an unsolicited
    frame carrying an opcode byte the protocol does not use is now withheld
    from the state callback like any other admission failure.
    """
    received: list[bytes] = []
    session = _make_session(received)
    frame = bytearray(RESPONSE_FRAME_LEN)
    frame[0x00] = 0xCC
    frame[0x03] = util._simple_checksum(frame)
    assert util._simple_checksum(frame) == 0

    session._notify(0, frame)

    assert received == []


@pytest.mark.asyncio
async def test_a_rejection_during_a_failing_write_is_not_left_unretrieved() -> None:
    """A wait failed mid-write must not leak an unretrieved exception.

    A frame can fail the wait while the GATT write itself is still in
    flight; when the write then raises, nothing ever awaits the future the
    ResponseError was set on. The write's own error still propagates, and
    the finally block retrieves the orphaned exception so asyncio does not
    log 'exception was never retrieved' later.
    """
    received: list[bytes] = []
    session = _make_session(received)
    orphaned: list[asyncio.Future[bytes]] = []

    async def deliver(*_args: object, **_kwargs: object) -> None:
        assert session._notify_future is not None
        orphaned.append(session._notify_future)
        session._notify(0, bytearray(9))
        raise BleakError("write failed")

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    with pytest.raises(BleakError, match="write failed"):
        await session.execute(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    # _log_traceback is the flag asyncio checks on GC to emit "exception was
    # never retrieved"; set_exception raises it and only a retrieval clears
    # it, so False here proves the finally block consumed the error. It must
    # be read before exception() below, which would clear it itself.
    (future,) = orphaned
    assert future._log_traceback is False
    assert isinstance(future.exception(), ResponseError)


@pytest.mark.asyncio
async def test_the_command_builders_emit_exactly_one_response_frame() -> None:
    """Both builders and the admission gate share the one frame length.

    The gate refuses anything that is not RESPONSE_FRAME_LEN bytes, so a
    command builder drifting from it would emit frames the link itself
    rejects.
    """
    session = _make_session([])
    cmd = session.build_operation_command(0x0B, 0x05)

    assert len(cmd) == RESPONSE_FRAME_LEN
    assert cmd[0x00] == 0xEE
    assert cmd[0x01] == 0x0B
    assert cmd[0x04] == 0x05
    assert cmd[0x10] == 0x02


SESSION_KEY = bytes(range(16))


def _lock_side_frames(*plaintexts: bytes) -> list[bytes]:
    """Encrypt frames the way the lock does, as one chained stream.

    The first 16 bytes are one CBC block and the last two travel in the clear,
    and the context chains across frames, which is why a frame that reaches
    our decryptor out of step desynchronizes every frame after it.
    """
    encryptor = Cipher(algorithms.AES(SESSION_KEY), modes.CBC(bytes(0x10))).encryptor()
    return [encryptor.update(p[0:0x10]) + p[0x10:0x12] for p in plaintexts]


@pytest.mark.asyncio
async def test_a_short_frame_is_dropped_before_it_can_poison_the_cipher() -> None:
    """A wrong-length payload never reaches the cipher, so the stream survives.

    The CBC context consumes ciphertext in 16-byte blocks. A partial block
    would stay buffered inside it and every later frame on the connection
    would decrypt misaligned, which only a reconnect's set_key could undo. The
    frame after the runt must still decode.
    """
    received: list[bytes] = []
    session = _make_session(received, key=SESSION_KEY)
    first, second = _lock_side_frames(READ_ACK, READ_ANSWER)

    session._notify(0, bytearray(first))
    session._notify(0, bytearray(b"\x00" * 10))
    session._notify(0, bytearray(second))

    assert received == [READ_ACK, READ_ANSWER]


@pytest.mark.asyncio
async def test_an_unsolicited_invalid_frame_is_withheld_from_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame arriving with no wait armed is validated too, and dropped.

    The state callback drives the consumer's state whether or not a command is
    outstanding, so an unsolicited frame has to clear the same bar.
    """
    received: list[bytes] = []
    session = _make_session(received)
    short_frame = bytearray(9)

    with caplog.at_level(logging.INFO, logger="yalexs_ble.session"):
        session._notify(0, bytearray(CORRUPT_ANSWER))
        session._notify(0, short_frame)

    assert received == []
    assert "dropping invalid frame" in caplog.text
    assert "9-byte payload is not an 18-byte response frame" in caplog.text
    # The drop line carries the frame itself, so a survey of the stream can
    # still read what was refused.
    assert CORRUPT_ANSWER.hex() in caplog.text
    assert short_frame.hex() in caplog.text


@pytest.mark.asyncio
async def test_a_short_frame_ends_an_armed_wait_and_the_command_is_re_sent() -> None:
    """A wrong-length frame reaches the waiter instead of costing it a timeout.

    The checksum helper indexes 18 bytes unconditionally, so before the gate a
    short frame raised IndexError out of the callback with the wait still
    armed, and the caller paid the whole response timeout before it could try
    again. The gate ends the wait with a ResponseError, so the command goes
    out again straight away.
    """
    received: list[bytes] = []
    session = _make_session(received)
    frames = [bytearray(9), bytearray(READ_ANSWER)]

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, frames.pop(0))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session.client.write_gatt_char.await_count == 2
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_an_empty_notification_leaves_an_armed_wait_armed() -> None:
    """An empty payload is dropped without failing the wait.

    The real response is still in flight when the empty notification lands;
    failing the wait would re-send the command while it is, and the lock
    would decrypt the duplicate as garbage. The wait stays armed and the
    real frame answers it, with no second write.
    """
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray())
        # The empty payload must not have failed the wait.
        assert session._notify_future is not None
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session.client.write_gatt_char.await_count == 1
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_a_frame_racing_a_cancelled_wait_does_not_raise() -> None:
    """A frame landing after the waiter gave up must not blow up the callback.

    A timeout or a disconnect cancels the armed future from the waiting side,
    and a notification already queued in that window then finds a future that
    is done. Resolving it again would raise InvalidStateError out of the
    notify callback; instead the wait is disarmed and the frame still takes
    its normal path.
    """
    received: list[bytes] = []
    session = _make_session(received)

    def arm_cancelled_wait() -> None:
        future: asyncio.Future[bytes] = session.loop.create_future()
        session._notify_future = future
        future.cancel()

    # A valid frame against a cancelled wait: state still updates.
    arm_cancelled_wait()
    session._notify(0, bytearray(READ_ANSWER))
    assert received == [READ_ANSWER]
    assert session._notify_future is None

    # An invalid frame against a cancelled wait: dropped, wait disarmed.
    arm_cancelled_wait()
    session._notify(0, bytearray(9))
    assert received == [READ_ANSWER]
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_an_over_length_frame_is_refused_and_logged_a_level_above(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The length gate is strict equality, and over-length earns more noise.

    A frame carrying more than the response length would otherwise validate
    on its first 18 bytes and be passed on with a tail the cipher never saw,
    so it is refused like a short one. It is also logged a level above: the
    radio truncates frames on its own, but nothing on the link builds a
    longer one, so over-length points at the transport below.
    """
    received: list[bytes] = []
    session = _make_session(received)
    assert len(READ_ANSWER) == RESPONSE_FRAME_LEN

    with caplog.at_level(logging.INFO, logger="yalexs_ble.session"):
        session._notify(0, bytearray(READ_ANSWER + READ_ANSWER))
        session._notify(0, bytearray(9))

    assert received == []
    drops = [r for r in caplog.records if "dropping invalid frame" in r.getMessage()]
    assert [r.levelname for r in drops] == ["WARNING", "INFO"]


def _secure_on_air(frame: bytes | bytearray) -> bytes:
    """Encrypt a frame the way the lock sends it on the secure channel.

    The first 16 bytes are one ECB block and the last two travel in the clear,
    which is the same shape as the plain channel with a different mode.
    """
    encryptor = Cipher(algorithms.AES(SESSION_KEY), modes.ECB()).encryptor()  # nosec
    return encryptor.update(bytes(frame[0x00:0x10])) + bytes(frame[0x10:0x12])


@pytest.mark.asyncio
async def test_a_secure_session_handshake_frame_still_answers_its_wait() -> None:
    """The authentication path shares _notify, so admission runs on it too.

    SecureSession overrides the cipher mode and the checksum rule but not
    _notify, so the length gate and the validation now ahead of the state
    callback both run on the handshake. A handshake reply has to clear them,
    and it carries no state callback to fall back on.
    """
    client = MagicMock(is_connected=True)
    session = SecureSession(client, "testlock", asyncio.Lock(), set(), 1)
    session.set_key(SESSION_KEY)
    # The lock's reply to the key exchange, checksummed as the lock stamps it.
    answer = session.build_command(0x02)
    session._write_checksum(answer)
    on_air = _secure_on_air(answer)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(on_air))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    assert len(on_air) == RESPONSE_FRAME_LEN
    result = await session.execute(session.build_command(0x01), "KEY_EXCHANGE")
    assert result == bytes(answer)
