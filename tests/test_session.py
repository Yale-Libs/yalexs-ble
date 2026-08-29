"""Session-level tests for frame admission, the response matcher and the staged wait."""

import asyncio
import logging
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak import BleakError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from yalexs_ble import util
from yalexs_ble.const import Commands, SettingType
from yalexs_ble.lock import (
    _ack_matcher,
    _operation_response_matcher,
    _settings_response_matcher,
)
from yalexs_ble.push import RETRYABLE_EXCEPTIONS, SLOW_TIMEOUT
from yalexs_ble.secure_session import SecureSession
from yalexs_ble.session import (
    ACK_TIMEOUT,
    COOLDOWN_TIME,
    OPERATION_RESPONSE_TIMEOUT,
    RESPONSE_FRAME_LEN,
    AuthError,
    DisconnectedError,
    OperationIncompleteError,
    OperationProgress,
    ResponseError,
    Session,
)

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
    previous command's matcher still in place. The staged operation wait skips
    corrupt frames instead; this is the plain path, where re-writing a settings
    command costs nothing.
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


def _with_checksum(hex_str: str) -> bytearray:
    """Build an 18-byte frame with a valid simple checksum in byte[3].

    _validate_response requires the 18-byte simple checksum to sum to zero;
    byte[3] is the checksum field, so set it to whatever makes that hold.
    """
    frame = bytearray.fromhex(hex_str)
    frame[0x03] = 0
    frame[0x03] = util._simple_checksum(frame)
    return frame


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Yield to the event loop until predicate() holds (bounded).

    Lets the operation under test advance to its next awaited stage before the
    test feeds the next frame, so each _notify lands in the intended stage.
    """
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


async def _spin_until_written(client: MagicMock, writes: int = 1) -> None:
    """Yield until the GATT write has returned.

    The write is where the staged wait begins: both stage futures are armed
    before it, and the writer reaches its first await straight after it. A
    frame fed once the write has returned therefore lands in the stage the
    test intends.
    """
    await _spin_until(lambda: client.write_gatt_char.await_count >= writes)


def _make_operation_session(
    state_callback: Callable[[bytes], None] | None = None,
) -> tuple[Session, MagicMock]:
    """Build a Session over a mock client for the notify/staged-wait path.

    Only cipher_encrypt is set (execute_operation asserts it); cipher_decrypt
    is left None so notify frames pass through Session.decrypt unmodified.
    """
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    session = Session(client, "mylock", asyncio.Lock(), set(), state_callback)
    session.cipher_encrypt = Cipher(
        algorithms.AES(bytes(16)),
        modes.CBC(bytes(16)),
    ).encryptor()
    return session, client


# securemode = LOCK opcode (0x0B) with operation byte 0x04.
_ACK_SECUREMODE = "aa0b00000400000000000000000000000200"
_OP_RESPONSE_OK = "bb0b00000000000000000000000000000200"
_SETTLED_STATUS = "bb0200000200000000000000000000000200"
_FOREIGN_ACK = "aa0b00000000000000000000000000000200"


@pytest.mark.asyncio
async def test_execute_operation_happy_path() -> None:
    """Write, then ack, then op-response: returns the op-response bytes.

    write_success_callback fires once, before the ack is delivered.
    """
    order: list[str] = []
    session, client = _make_operation_session()
    write_cb = MagicMock(side_effect=lambda: order.append("write"))
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        order.append("ack")
        session._notify(0, bytearray(ack))
        # Deliver the op-response in the SAME event-loop turn: both stage
        # futures resolve before the writer resumes, exercising the
        # supersede branch, which must still record the acknowledgment.
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
        write_success_callback=write_cb,
    )
    await feeder

    assert result == bytes(op_response)
    assert client.write_gatt_char.await_count == 1
    assert progress.write_attempted is True
    assert progress.acknowledged is True
    write_cb.assert_called_once()
    # The callback fired before the ack was delivered.
    assert order == ["write", "ack"]


@pytest.mark.asyncio
async def test_a_raising_write_success_callback_does_not_abort_the_staged_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A write_success_callback that raises is contained, and the wait goes on.

    The hook runs after the command is confirmed delivered, so an exception
    escaping it would abandon a wait whose motor may be running, and the
    escaped type would be retried upstream and re-send the command. The
    exception is logged at error level, since a raising hook is a bug in the
    caller, and the staged wait still completes on its op-response.
    """
    session, client = _make_operation_session()
    write_cb = MagicMock(side_effect=RuntimeError("hook bug"))
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    with caplog.at_level("ERROR", logger="yalexs_ble.session"):
        result = await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
            write_success_callback=write_cb,
        )
    await feeder

    assert result == bytes(op_response)
    assert progress.acknowledged is True
    write_cb.assert_called_once()
    assert "write success callback for force_securemode raised" in caplog.text
    assert "hook bug" in caplog.text


@pytest.mark.asyncio
async def test_execute_operation_ignores_foreign_frames() -> None:
    """A settle and a foreign ack mid-wait do not complete the operation."""
    seen: list[bytes] = []
    session, client = _make_operation_session(seen.append)
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    settled = _with_checksum(_SETTLED_STATUS)
    foreign_ack = _with_checksum(_FOREIGN_ACK)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        # A settled GETSTATUS (0xBB 0x02) and a foreign ack (0xAA 0x0B with
        # operation byte 0x00, not the securemode 0x04) must not complete it.
        session._notify(0, bytearray(settled))
        await asyncio.sleep(0)
        session._notify(0, bytearray(foreign_ack))
        await asyncio.sleep(0)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.acknowledged is True
    # The foreign frames were still handed to the state callback.
    assert bytes(settled) in seen
    assert bytes(foreign_ack) in seen


@pytest.mark.asyncio
async def test_a_repeated_acknowledgment_is_inert_and_the_operation_completes() -> None:
    """A second matching acknowledgment resolves nothing and raises nothing.

    The acknowledgment slot is cleared as the first matching frame resolves
    it, which is what makes the resolution safe without a done() check. Left
    armed, the duplicate would resolve a future that is already done and raise
    InvalidStateError inside the notify callback, which the Bluetooth stack
    calls with nobody to catch it.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)
    escaped: list[BaseException] = []

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        try:
            # The same acknowledgment again, now in the op-response stage.
            session._notify(0, bytearray(ack))
        except Exception as err:
            escaped.append(err)
        await asyncio.sleep(0)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert escaped == []
    assert result == bytes(op_response)
    assert progress.acknowledged is True


@pytest.mark.asyncio
async def test_operation_wait_skips_a_corrupt_frame_and_completes() -> None:
    """A frame that fails the checksum mid-operation is skipped, not surfaced.

    On the plain path a checksum failure ends the wait so the caller can write
    the command again. A mechanical command must not be written again on the
    strength of a bad frame, so the operation wait skips it and keeps waiting
    for the op-response.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)
    # The op-response frame with its checksum field broken, so it decodes as
    # this command's answer but fails validation.
    corrupt = bytearray(op_response)
    corrupt[0x03] ^= 0xFF

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(corrupt))
        await asyncio.sleep(0)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    # The corrupt frame neither answered the command nor ended the wait.
    assert result == bytes(op_response)
    assert session.client.write_gatt_char.await_count == 1


@pytest.mark.asyncio
async def test_operation_wait_skips_a_corrupt_frame_after_the_acknowledgment() -> None:
    """A corrupt frame in the op-response stage is skipped too.

    The acknowledgment's own arming is cleared the moment it arrives, so the
    op-response stage looks exactly like a plain wait; _operation_progress is
    what still marks it as an operation. Keying the skip on the acknowledgment
    arming instead would error this wait and re-send a mechanical command the
    lock has already taken.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)
    corrupt = bytearray(op_response)
    corrupt[0x03] ^= 0xFF

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        session._notify(0, bytearray(corrupt))
        await asyncio.sleep(0)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert session.client.write_gatt_char.await_count == 1


@pytest.mark.asyncio
async def test_operation_progress_is_cleared_so_the_plain_path_still_retries() -> None:
    """The staged wait clears its progress record on the way out.

    _operation_progress is what tells _notify to skip a corrupt frame rather
    than error the wait. Left set after an operation, it would disable the
    plain path's corrupt-frame retry for the life of the connection.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    await session.execute_operation(
        session.build_operation_command(Commands.LOCK, 0x04),
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert session._operation_progress is None

    # A plain command behind the operation: its corrupt frame must still end
    # the wait and make _locked_write write the command again.
    good = _with_checksum(_SETTLED_STATUS)
    bad = bytearray(good)
    bad[0x03] ^= 0xFF
    frames = [bad, good]

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(frames.pop(0)))

    client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(
        session.build_command(Commands.GETSTATUS), "lock_status"
    )

    assert result == bytes(good)
    assert client.write_gatt_char.await_count == 2


@pytest.mark.asyncio
async def test_execute_operation_op_response_before_ack() -> None:
    """The op-response alone completes the wait; the ack stage is superseded."""
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    # No acknowledgment was needed to complete.
    assert progress.acknowledged is False


@pytest.mark.asyncio
async def test_a_stale_same_opcode_op_response_completes_a_fresh_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A previous same-opcode command's late op-response is taken as the answer.

    An op-response is matched on the opcode alone, so a fresh wait cannot
    tell its own answer from the previous same-opcode command's late one:
    fed during stage 1, the stale frame completes the wait through the
    supersede branch. progress.acknowledged staying False is the tell that
    the result may not be this command's, and the completion is logged at
    info so the case is visible in a field log.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    # The same bytes a previous lock-opcode command's op-response carries.
    stale_op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        # Arrives before this command's acknowledgment.
        session._notify(0, bytearray(stale_op_response))

    feeder = asyncio.create_task(feed())
    with caplog.at_level("INFO", logger="yalexs_ble.session"):
        result = await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder

    assert result == bytes(stale_op_response)
    assert progress.acknowledged is False
    assert "completed on its op-response; no acknowledgment was received" in caplog.text


@pytest.mark.asyncio
async def test_execute_operation_ack_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No acknowledgment within the ack budget raises a plain TimeoutError."""
    monkeypatch.setattr("yalexs_ble.session.ACK_TIMEOUT", 0.05)
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)

    with pytest.raises(TimeoutError) as exc_info:
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )

    # A plain TimeoutError, not the non-retryable OperationIncompleteError:
    # nothing was acknowledged, so the command may still be retried.
    assert not isinstance(exc_info.value, OperationIncompleteError)
    assert client.write_gatt_char.await_count == 1
    assert progress.acknowledged is False


def test_ack_timeout_outlasts_the_link_supervision_timeout() -> None:
    """Stage 1 must outlast a dead link, so a dead link reads as a disconnect.

    SLOW_TIMEOUT is the slow-mode supervision timeout in BLE units of 10 ms.
    Below it, a link that has already gone expires this stage instead, and a
    missing acknowledgment is retryable: the command would be written again
    to a lock that may have taken it.
    """
    assert ACK_TIMEOUT > SLOW_TIMEOUT / 100


def test_operation_response_timeout_outlasts_the_acknowledgment_budget() -> None:
    """Stage 2's budget must exceed stage 1's.

    Both stage deadlines run from the command issue, so everything the
    acknowledgment stage consumes comes out of the op-response budget; a
    budget at or below ACK_TIMEOUT could reach stage 2 with nothing left and
    fail the operation the moment the acknowledgment lands. The relationship
    is the invariant; the value itself is sized off the observed op-response
    arrival distribution and moves with it.
    """
    assert OPERATION_RESPONSE_TIMEOUT > ACK_TIMEOUT


def test_operation_incomplete_error_passes_the_retry_decorator() -> None:
    """The type's whole purpose is to end the attempt ladder.

    Its docstring says it is deliberately outside the bleak retry set, and
    the consequence of losing that is a mechanical command re-sent after its
    result went missing. Reparenting it under ResponseError would do exactly
    that and leaves every other test in this file green.
    """
    assert not issubclass(OperationIncompleteError, RETRYABLE_EXCEPTIONS)


@pytest.mark.asyncio
async def test_execute_operation_op_response_timeout() -> None:
    """Acknowledged but no op-response raises the non-retryable error."""
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=0.1,
            progress=progress,
        )
    await feeder

    assert progress.acknowledged is True


@pytest.mark.asyncio
async def test_stage_deadlines_run_from_the_command_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stage timeouts subtract the time already spent in the attempt.

    The acknowledgment wait asks for ACK_TIMEOUT minus the time the
    write consumed, and the op-response wait asks for response_timeout minus
    everything spent up to the acknowledgment, so the whole operation is
    bounded from the moment the command is issued, exactly as
    OperationIncompleteError's message states.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    clock = {"now": 1000.0}
    monkeypatch.setattr("yalexs_ble.session.monotonic", lambda: clock["now"])

    async def write_taking_half_a_second(*args: object, **kwargs: object) -> None:
        clock["now"] += 0.5

    client.write_gatt_char = AsyncMock(side_effect=write_taking_half_a_second)

    ack_wait_timeouts: list[float | None] = []
    real_wait = asyncio.wait

    async def spying_wait(fs: object, **kwargs: object) -> object:
        ack_wait_timeouts.append(kwargs.get("timeout"))  # type: ignore[arg-type]
        return await real_wait(fs, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("yalexs_ble.session.asyncio.wait", spying_wait)

    op_wait_timeouts: list[float] = []
    real_asyncio_timeout = util.asyncio_timeout

    def spying_asyncio_timeout(delay: float) -> object:
        op_wait_timeouts.append(delay)
        return real_asyncio_timeout(delay)

    monkeypatch.setattr(
        "yalexs_ble.session.util.asyncio_timeout", spying_asyncio_timeout
    )

    async def feed() -> None:
        await _spin_until_written(client)
        clock["now"] += 2.5  # the acknowledgment lands 3.0 s into the attempt
        session._notify(0, bytearray(ack))
        # The op-response must land in the op-response stage, so wait for that
        # stage to arm its own bound rather than for progress.acknowledged,
        # which is set as the frame is received and so is already true here.
        await _spin_until(lambda: len(op_wait_timeouts) == 2)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=12.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    # The write consumed 0.5 s of the attempt, so the acknowledgment stage
    # asked for the remainder of ACK_TIMEOUT, not the full budget again.
    assert ack_wait_timeouts == [pytest.approx(ACK_TIMEOUT - 0.5)]
    # The first bound is the write stage's own, ACK_TIMEOUT. The
    # acknowledgment then spent 3.0 s of the 12.0 s response budget, so the
    # op-response stage asked for the 9.0 s remainder.
    assert op_wait_timeouts == [
        pytest.approx(ACK_TIMEOUT),
        pytest.approx(9.0),
    ]


@pytest.mark.asyncio
async def test_execute_with_response_matcher_skips_nonmatching_and_bad_frames() -> None:
    """The plain execute path waits past a non-matching and a garbled frame."""
    session, _ = _make_operation_session()
    command = session.build_command(Commands.GETSTATUS)

    def matcher(data: bytes) -> bool:
        return len(data) > 1 and data[0] == 0xBB and data[1] == 0x02

    nonmatching = _with_checksum(_OP_RESPONSE_OK)  # valid 0xBB 0x0B, wrong op
    good = _with_checksum(_SETTLED_STATUS)  # valid 0xBB 0x02, matches
    bad = _with_checksum(_SETTLED_STATUS)
    bad[0x03] = (bad[0x03] + 1) & 0xFF  # corrupt the checksum

    async def feed() -> None:
        await _spin_until(lambda: session._notify_future is not None)
        session._notify(0, bytearray(nonmatching))
        await asyncio.sleep(0)
        # A garbled frame during a plain wait DOES end it, matcher or no
        # matcher: the wait is errored and _locked_write writes the command
        # again. Only a staged operation wait skips corrupt frames.
        session._notify(0, bytearray(bad))
        await asyncio.sleep(0)
        session._notify(0, bytearray(good))

    feeder = asyncio.create_task(feed())
    result = await session.execute(command, "lock_status", matcher)
    await feeder

    assert result == bytes(good)
    assert session.client.write_gatt_char.await_count == 2


@pytest.mark.asyncio
async def test_cooldown_paces_a_command_behind_the_last_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the cooldown enabled, a command waits out the rest of COOLDOWN_TIME.

    Both write paths share this wait. A command sent too soon after a frame
    can stop the lock advertising, which needs a battery pull to recover, so
    the session paces itself off the last frame it saw.
    """
    session, _ = _make_operation_session()
    session.enable_cooldown()
    clock = {"now": 1000.0}
    monkeypatch.setattr("yalexs_ble.session.monotonic", lambda: clock["now"])
    session._last_callback_time = clock["now"] - 0.2
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)
        clock["now"] += delay

    monkeypatch.setattr("yalexs_ble.session.asyncio.sleep", fake_sleep)
    await session._wait_for_cooldown()

    assert slept == [pytest.approx(COOLDOWN_TIME - 0.2)]


@pytest.mark.asyncio
async def test_execute_operation_pays_the_cooldown_before_its_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mechanical command waits out the cooldown before it is written.

    A command written too soon after the lock's last frame can stop the lock
    advertising, so the operation path pays the same wait as the settings
    path rather than writing straight away.
    """
    session, client = _make_operation_session()
    session.enable_cooldown()
    clock = {"now": 1000.0}
    monkeypatch.setattr("yalexs_ble.session.monotonic", lambda: clock["now"])
    session._last_callback_time = clock["now"] - 0.2
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        if delay:
            slept.append(delay)
            clock["now"] += delay
        # The cooldown's own wait is the one under test; every other sleep in
        # this test is a yield, so hand the loop a real one.
        await real_sleep(0)

    monkeypatch.setattr("yalexs_ble.session.asyncio.sleep", fake_sleep)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        session.build_operation_command(Commands.LOCK, 0x04),
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=OperationProgress(),
    )
    await feeder

    assert result == bytes(op_response)
    assert slept == [pytest.approx(COOLDOWN_TIME - 0.2)]


# =========================================================================== #
# Disconnect / auth contract of the operation wait
#
# The pre/post-acknowledgment retry hinge: a failure BEFORE the ack keeps its
# retryable type (the command never moved the motor); once acknowledged the
# result is unknown, so a disconnect or timeout becomes the non-retryable
# OperationIncompleteError and the command is never silently re-sent. These
# tests pin that contract so a future change that makes an acknowledged
# operation retryable again fails the suite instead of shipping.
# =========================================================================== #
def _fire_disconnect(session: Session) -> None:
    """Resolve the operation's disconnected future, as a link drop would.

    execute_operation registers a future in _disconnected_futures and wraps the
    wait in async_interrupt.interrupt(...); resolving that future raises
    DisconnectedError inside the wait.
    """
    for fut in list(session._disconnected_futures):
        if not fut.done():
            fut.set_result(None)


@pytest.mark.asyncio
async def test_execute_operation_disconnect_after_ack_is_operation_incomplete() -> None:
    """Disconnect AFTER the ack -> non-retryable OperationIncompleteError."""
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        _fire_disconnect(session)  # link drops while awaiting the op-response

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder


@pytest.mark.asyncio
async def test_ack_and_disconnect_in_one_turn_is_operation_incomplete() -> None:
    """Ack and disconnect in ONE event-loop turn -> non-retryable.

    The interrupt cancels the staged wait before it resumes, so nothing in the
    wait body observes the acknowledgment. It still has to be recorded, or the
    drop is classified as a pre-acknowledgment disconnect and the command is
    re-sent to a lock that already took it.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until_written(client)
        # Both resolutions land in the same turn, with no await between them.
        session._notify(0, bytearray(ack))
        _fire_disconnect(session)

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder

    assert progress.acknowledged is True


@pytest.mark.asyncio
async def test_op_response_and_disconnect_in_one_turn_returns_the_result() -> None:
    """Op-response and disconnect in ONE event-loop turn -> the result stands.

    The twin of the acknowledgment race, with the opposite consequence: the
    interrupt cancels the staged wait before it resumes, so the wait body never
    sees the op-response it was given. Recorded where the frame arrives, the
    result is in hand and a completed operation is reported as completed
    instead of as one whose result never arrived.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        # Both land in the same turn, with no await between them.
        session._notify(0, bytearray(op_response))
        _fire_disconnect(session)

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.result == bytes(op_response)


@pytest.mark.asyncio
async def test_op_response_and_stage_two_timeout_in_one_turn_returns_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Op-response and stage-2 timeout in ONE event-loop turn -> the result stands.

    The third twin, with the timer in the disconnect's role: the notify
    delivering the op-response and the stage-2 timeout handle can become due
    in the same event-loop iteration, and the timeout then cancels the wait
    before it can resume with the result the future already holds. Recorded
    where the frame arrives, the result is in hand and is returned instead of
    raising OperationIncompleteError.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    op_wait_timeouts: list[float] = []
    real_asyncio_timeout = util.asyncio_timeout

    def spying_asyncio_timeout(delay: float) -> object:
        op_wait_timeouts.append(delay)
        return real_asyncio_timeout(delay)

    monkeypatch.setattr(
        "yalexs_ble.session.util.asyncio_timeout", spying_asyncio_timeout
    )

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        # Wait for the op-response stage to arm its own timer, so the blocking
        # sleep below expires that timer and not the acknowledgment stage's.
        await _spin_until(lambda: len(op_wait_timeouts) == 2)
        # Key the schedule to the armed stage-2 budget, so the ordering
        # holds however long the run spent getting here.
        armed = op_wait_timeouts[1]
        # armed == 0 would put the notify and the timeout on one deadline,
        # decided by scheduling order; a stalled run must fail as stalled.
        assert armed > 0, "the fixture stalled past the stage-2 budget"
        loop = asyncio.get_running_loop()
        loop.call_later(armed / 4, session._notify, 0, bytearray(op_response))
        # Block the loop past both deadlines: when it wakes, the notify handle
        # (the earlier deadline) and the stage-2 timeout handle run in the
        # same iteration, notify first, reproducing the race. The blocking
        # sleep is the point, so the async lint rule is suppressed.
        time.sleep(armed + 0.1)  # noqa: ASYNC251

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=0.2,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.result == bytes(op_response)


@pytest.mark.asyncio
async def test_an_op_response_behind_the_stage_two_timeout_returns_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The op-response due after the stage-2 deadline still stands.

    The twin of the ordering above: the two handles become due in one
    event-loop iteration with the deadline the earlier of them, so the wait is
    cancelled first and the frame lands on a wait that has not yet resumed.
    Recorded where the frame arrives, the result is in hand either way round.
    """
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    op_wait_timeouts: list[float] = []
    real_asyncio_timeout = util.asyncio_timeout

    def spying_asyncio_timeout(delay: float) -> object:
        op_wait_timeouts.append(delay)
        return real_asyncio_timeout(delay)

    monkeypatch.setattr(
        "yalexs_ble.session.util.asyncio_timeout", spying_asyncio_timeout
    )

    async def feed() -> None:
        await _spin_until_written(client)
        session._notify(0, bytearray(ack))
        # Wait for the op-response stage to arm its own timer, so the blocking
        # sleep below expires that timer and not the acknowledgment stage's.
        await _spin_until(lambda: len(op_wait_timeouts) == 2)
        # Key the schedule to the armed stage-2 budget, so the ordering
        # holds however long the run spent getting here.
        armed = op_wait_timeouts[1]
        # armed == 0 would put the notify and the timeout on one deadline,
        # decided by scheduling order; a stalled run must fail as stalled.
        assert armed > 0, "the fixture stalled past the stage-2 budget"
        loop = asyncio.get_running_loop()
        loop.call_later(armed + 0.05, session._notify, 0, bytearray(op_response))
        # Block the loop past both deadlines: when it wakes, the stage-2
        # timeout handle (the earlier deadline) and the notify handle run in
        # the same iteration, the timeout first, which is the ordering this
        # test pins. The blocking sleep is the point, so the async lint rule
        # is suppressed.
        time.sleep(armed + 0.15)  # noqa: ASYNC251

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=0.2,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.result == bytes(op_response)


@pytest.mark.asyncio
async def test_execute_operation_disconnect_before_ack_stays_retryable() -> None:
    """Disconnect BEFORE the ack -> plain DisconnectedError (retryable)."""
    session, client = _make_operation_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)

    async def feed() -> None:
        await _spin_until_written(client)
        _fire_disconnect(session)  # link drops before any acknowledgment

    feeder = asyncio.create_task(feed())
    with pytest.raises(DisconnectedError) as exc_info:
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder

    # Retryable: not the non-retryable type, and nothing was acknowledged.
    assert not isinstance(exc_info.value, OperationIncompleteError)
    assert progress.acknowledged is False


@pytest.mark.asyncio
async def test_execute_operation_key_error_on_write_raises_auth_error() -> None:
    """A key/slot error on the first request surfaces as AuthError."""
    session, client = _make_operation_session()
    client.write_gatt_char = AsyncMock(side_effect=BleakError("Unlikely Error"))

    with pytest.raises(AuthError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=OperationProgress(),
        )


@pytest.mark.asyncio
async def test_execute_operation_not_connected_raises_disconnected() -> None:
    """The pre-write connected-guard raises, converted to DisconnectedError.

    The one exit taken before the write call, so it is where write_attempted
    stays False: a caller that must never re-send a command reads that
    record, and every later exit has to report the command as possibly
    delivered.
    """
    session, client = _make_operation_session()
    client.is_connected = False
    progress = OperationProgress()

    with pytest.raises(DisconnectedError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )

    assert progress.write_attempted is False


@pytest.mark.asyncio
async def test_errored_write_still_reports_the_write_as_attempted() -> None:
    """An error raised by the write call leaves write_attempted True.

    The request PDU can leave the radio with only the ATT response lost, so
    an error from the write leaves delivery unknown, and write_attempted is
    what carries that to the caller: set after the call instead, an errored
    write would report False and read as a command that was never sent.
    """
    session, client = _make_operation_session()
    client.write_gatt_char.side_effect = BleakError("write failed")
    progress = OperationProgress()

    with pytest.raises(BleakError, match="write failed"):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )

    assert progress.write_attempted is True


@pytest.mark.asyncio
async def test_bleak_disconnect_after_ack_is_operation_incomplete() -> None:
    """Defensive branch: a BleakError disconnect surfacing AFTER the ack is also
    treated as an unknown result (OperationIncompleteError), not a retry."""
    session, _ = _make_operation_session()
    progress = OperationProgress()

    def _boom(
        command: bytearray,
        command_name: str,
        ack_matcher: Callable[[bytes], bool],
        response_matcher: Callable[[bytes], bool],
        response_timeout: float,
        progress: OperationProgress,
        write_success_callback: Callable[[], None] | None = None,
    ) -> bytes:
        progress.acknowledged = True
        raise BleakError("device disconnected")

    with (
        patch.object(session, "_locked_write_operation", AsyncMock(side_effect=_boom)),
        pytest.raises(OperationIncompleteError),
    ):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )


@pytest.mark.asyncio
async def test_bleak_disconnect_in_one_turn_with_the_op_response_returns_it() -> None:
    """The same twin on the BleakError arm: a recorded result still stands."""
    session, _ = _make_operation_session()
    progress = OperationProgress()
    op_response = bytes(_with_checksum(_OP_RESPONSE_OK))

    def _boom(
        command: bytearray,
        command_name: str,
        ack_matcher: Callable[[bytes], bool],
        response_matcher: Callable[[bytes], bool],
        response_timeout: float,
        progress: OperationProgress,
        write_success_callback: Callable[[], None] | None = None,
    ) -> bytes:
        progress.acknowledged = True
        progress.result = op_response
        raise BleakError("device disconnected")

    with patch.object(session, "_locked_write_operation", AsyncMock(side_effect=_boom)):
        result = await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )

    assert result == op_response


@pytest.mark.asyncio
async def test_execute_settings_not_connected_raises_disconnected() -> None:
    """The settings/poll path's pre-write connected-guard also raises."""
    session, client = _make_operation_session()
    client.is_connected = False

    with pytest.raises(DisconnectedError):
        await session.execute(session.build_command(Commands.GETSTATUS), "status")


@pytest.mark.asyncio
async def test_notify_returns_when_only_ack_is_armed_and_frame_does_not_match() -> None:
    """A frame the acknowledgment matcher refuses resolves nothing.

    The operation record and the acknowledgment wait are both armed, so the
    matcher is consulted, and with no result wait behind it the frame is
    dropped once the state callback has seen it.
    """
    session, _ = _make_operation_session()
    # The record is what admits a frame to the acknowledgment branch at all;
    # without it the matcher below is never reached.
    session._operation_progress = OperationProgress()
    session._ack_future = asyncio.get_running_loop().create_future()
    matcher = _ack_matcher(0x0B, 0x04)
    offered: list[bytes] = []

    def counting_matcher(data: bytes) -> bool:
        offered.append(data)
        return matcher(data)

    session._ack_matcher = counting_matcher
    session._notify_future = None
    session._notify_matcher = None

    # A settled GETSTATUS (0xBB) never matches the 0xAA ack matcher.
    frame = _with_checksum(_SETTLED_STATUS)
    session._notify(0, bytearray(frame))

    assert offered == [bytes(frame)]
    assert not session._ack_future.done()


@pytest.mark.asyncio
async def test_execute_operation_generic_bleak_error_is_reraised() -> None:
    """A BleakError that is neither a key nor a disconnect error re-raises as-is.

    It stays a plain (retryable) BleakError, not converted to AuthError or
    DisconnectedError, so the write may simply be retried.
    """
    session, client = _make_operation_session()
    client.write_gatt_char = AsyncMock(side_effect=BleakError("GATT write failed"))

    with pytest.raises(BleakError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=OperationProgress(),
        )


# =========================================================================== #
# 1000 ms field replays: the typed staged wait vs the wrong-capture frames
#
# Verbatim, ordered field frame sequences in which the OLD untyped wait
# captured a wrong frame, written ahead of the staged wait that answers them.
# Each is replayed through session._notify during an execute_operation staged
# wait. Frame classes:
#   * bb0a / bb0b = 0xBB op-response (opcode 0x0a unlock / 0x0b lock+securemode)
#   * bb02        = 0xBB 0x02 settled GETSTATUS status push
#   * aa0a / aa0b = 0xAA acknowledgment matching the written opcode
# byte[3] checksums are valid as captured, so the frames are fed unchanged.
# =========================================================================== #
@pytest.mark.asyncio
async def test_replay_incident1_force_lock_ignores_prev_unlock_stale_bb0a() -> None:
    """Incident 1: a force_lock wait ignores the previous unlock's stale bb0a.

    The old untyped wait was satisfied by bb0a003b..., the delayed op-response
    (opcode 0x0a = unlock) of the PRECEDING force_unlock, a leftover frame, not
    an answer to this force_lock (opcode 0x0b). The typed wait keeps it on the
    state-callback path and completes only on this force_lock's own bb0b.
    """
    seen: list[bytes] = []
    session, client = _make_operation_session(seen.append)
    progress = OperationProgress()
    command = session.build_command(Commands.LOCK)  # force_lock: opcode 0x0b

    # Verbatim field frames (byte[3] checksums valid as captured).
    stale_unlock_result = bytes.fromhex("bb0a003b0000000000000000000000000000")
    settled_unlocked = bytes.fromhex("bb02003e0200000003000000000000000000")
    true_lock_ack = bytes.fromhex("aa0b00490000000000000000000000000200")
    genuine_lock_result = bytes.fromhex("bb0b003a0000000000000000000000000000")

    op_task: asyncio.Task[bytes] | None = None

    async def feed() -> None:
        await _spin_until_written(client)
        # The previous unlock's stale op-response and a settled status push:
        # neither is this force_lock's ack or op-response.
        session._notify(0, bytearray(stale_unlock_result))
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert op_task is not None and not op_task.done()  # wrong frame ignored
        # The genuine LOCK ack (opcode 0x0b, op byte 0x00) advances to stage 2.
        session._notify(0, bytearray(true_lock_ack))
        await _spin_until(lambda: progress.acknowledged)
        # Only its own bb0b op-response completes the wait.
        session._notify(0, bytearray(genuine_lock_result))

    feeder = asyncio.create_task(feed())
    op_task = asyncio.create_task(
        session.execute_operation(
            command,
            "force_lock",
            ack_matcher=_ack_matcher(Commands.LOCK, 0x00),
            response_matcher=_operation_response_matcher(Commands.LOCK),
            response_timeout=5.0,
            progress=progress,
        )
    )
    result = await op_task
    await feeder

    assert result == genuine_lock_result
    # It went through the real ack stage; the stale bb0a did not short-circuit.
    assert progress.acknowledged is True
    # The stale op-response and the settle stayed on the state-callback path.
    assert stale_unlock_result in seen
    assert settled_unlocked in seen


@pytest.mark.asyncio
async def test_a_settled_status_push_completes_neither_operation_stage() -> None:
    """A settled status push answers neither stage of an unlock's staged wait.

    A GETSTATUS status push (0xBB 0x02) is not an operation frame, so it
    answers neither the acknowledgment stage nor the op-response stage. Only
    the unlock's own bb0a completes the wait.
    """
    seen: list[bytes] = []
    session, client = _make_operation_session(seen.append)
    progress = OperationProgress()
    command = session.build_command(Commands.UNLOCK)  # force_unlock: opcode 0x0a

    # Verbatim field frames (byte[3] checksums valid as captured).
    settled_unlocked = bytes.fromhex("bb02003e0200000003000000000000000000")
    true_unlock_ack = bytes.fromhex("aa0a004a0000000000000000000000000200")
    genuine_unlock_result = bytes.fromhex("bb0a003b0000000000000000000000000000")

    op_task: asyncio.Task[bytes] | None = None

    async def feed() -> None:
        await _spin_until_written(client)
        # A settled status push during the ack stage does not complete it.
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert op_task is not None and not op_task.done()
        session._notify(0, bytearray(true_unlock_ack))
        await _spin_until(lambda: progress.acknowledged)
        # A settled status push during the op-response stage does not either.
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert not op_task.done()
        session._notify(0, bytearray(genuine_unlock_result))

    feeder = asyncio.create_task(feed())
    op_task = asyncio.create_task(
        session.execute_operation(
            command,
            "force_unlock",
            ack_matcher=_ack_matcher(Commands.UNLOCK, 0x00),
            response_matcher=_operation_response_matcher(Commands.UNLOCK),
            response_timeout=5.0,
            progress=progress,
        )
    )
    result = await op_task
    await feeder

    assert result == genuine_unlock_result
    assert progress.acknowledged is True
    # Both settled pushes stayed on the state-callback path, never claimed.
    assert seen.count(settled_unlocked) == 2
