import asyncio
import contextlib
import copy
from collections.abc import Callable, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from bleak_retry_connector import BLEDevice
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import yalexs_ble
from yalexs_ble.const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
    MECHANICAL_OPERATION_ERRORS,
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    VALUE_TO_LOCK_STATUS,
    AutoLockMode,
    AutoLockState,
    Commands,
    LockInfo,
    LockOperationRemoteType,
    LockOperationSource,
    LockStateValue,
    LockStatus,
    OperationError,
    SettingType,
)
from yalexs_ble.lock import (
    AA_BATTERY_VOLTAGE_TO_PERCENTAGE,
    UNLATCH_OPERATION_BYTE,
    Lock,
    _ack_matcher,
    _operation_response_matcher,
    _settings_response_matcher,
    convert_voltage_to_percentage,
)
from yalexs_ble.session import (
    OPERATION_RESPONSE_TIMEOUT,
    UNLATCH_OPERATION_RESPONSE_TIMEOUT,
    DisconnectedError,
    OperationFailedError,
    OperationIncompleteError,
    OperationProgress,
    Session,
    UnlatchError,
)
from yalexs_ble.util import _simple_checksum


def test_aa_battery_voltage_to_percentage_is_monotonic() -> None:
    """Percentage must be non-increasing as voltage decreases.

    Guards against copy/paste regressions in the lookup table — a non-monotonic
    table makes ``convert_voltage_to_percentage`` return higher percentages for
    lower voltages, which erodes user trust in the battery indicator.
    """
    sorted_pairs = sorted(AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    percents = [pct for _, pct in sorted_pairs]
    assert percents == sorted(percents), (
        f"voltage→pct table is non-monotonic: {sorted_pairs}"
    )


def test_convert_voltage_to_percentage_is_monotonic_across_table() -> None:
    """``convert_voltage_to_percentage`` must be non-decreasing in voltage."""
    voltages = sorted(v for v, _ in AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    results = [convert_voltage_to_percentage(v) for v in voltages]
    assert results == sorted(results), (
        f"convert_voltage_to_percentage is non-monotonic across table voltages: "
        f"{list(zip(voltages, results, strict=True))}"
    )


def test_create_lock() -> None:
    Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )


@pytest.mark.asyncio
async def test_connection_canceled_on_disconnect() -> None:
    disconnect_mock = AsyncMock()
    mock_client = MagicMock(connected=True, disconnect=disconnect_mock)
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", delegate=""),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )
    lock.client = mock_client

    async def connect_and_wait() -> None:
        await lock.connect()
        await asyncio.sleep(2)

    with patch("yalexs_ble.lock.Lock.connect"):
        task = asyncio.create_task(connect_and_wait())
        await asyncio.sleep(0)
        task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.cancelled() is True


def test_parse_operation_source() -> None:
    """Test parsing operation source and remote type."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Test remote source with BLE type
    source, remote_type = lock._parse_operation_source(0x00, 0x03)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.BLE

    # Test manual source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x01, 0x03)
    assert source is LockOperationSource.MANUAL
    assert remote_type is None

    # Test auto lock source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x05, 0x00)
    assert source is LockOperationSource.AUTO_LOCK
    assert remote_type is None

    # Test PIN source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x0B, 0x03)
    assert source is LockOperationSource.PIN
    assert remote_type is None

    # Test unknown source
    source, remote_type = lock._parse_operation_source(0x99, 0x03)
    assert source is LockOperationSource.UNKNOWN
    assert remote_type is None

    # Test remote source with unknown remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x99)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN

    # Test remote source with UNKNOWN (0x00) remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x00)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN


def test_parse_lock_command_response_jammed() -> None:
    """LOCK op-response with a MECH_* result (byte[15]) parses as JAMMED."""
    lock = _make_lock()

    # Real lock-jam capture: byte[15] = 0x1F MECH_POSITION. byte[3] (0x1B
    # here) is only the frame checksum, not a status.
    frame = bytes.fromhex("bb0b001b00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_unlock_command_response_jammed() -> None:
    """UNLOCK op-response with a MECH_* result (byte[15]) parses as JAMMED.

    The old byte[3] path missed this: an unlock jam's checksum is 0x1C, not
    the 0x1B it looked for. The result is in byte[15] (0x1F MECH_POSITION)
    regardless of direction.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a001c00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_lock_command_response_success_is_no_update() -> None:
    """A successful LOCK op-response (byte[15]=0x00) carries no state update.

    The op-response reports the result of the issued command; which state
    resulted is known to the command issuer, not the parser (lock and
    securemode op-responses are byte-identical), so the parser emits nothing.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0b003a0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_unlock_command_response_success_is_no_update() -> None:
    """A successful UNLOCK op-response (byte[15]=0x00) carries no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0a003b0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_getstatus_staticposition() -> None:
    """A settled GETSTATUS lock state of 0x07 (STATICPOSITION) parses as JAMMED."""
    lock = _make_lock()

    # bb02 GETSTATUS, byte[4]=0x02 LOCK_ONLY, byte[8]=0x07 (settled jam state).
    frame = bytes.fromhex("bb02003a0200000007000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


@pytest.mark.parametrize(
    ("frame_hex", "expected"),
    [
        ("bb0200380200000009000000000000000000", LockStatus.UNLATCHING),
        ("bb020037020000000a000000000000000000", LockStatus.UNLATCHED),
    ],
    ids=["unlatching", "unlatched"],
)
def test_parse_getstatus_unlatch_states(frame_hex: str, expected: LockStatus) -> None:
    """A pushed GETSTATUS position of 0x09 or 0x0A decodes to the unlatch states.

    Both decoded as UNKNOWN while the two members were commented out of
    LockStatus, since VALUE_TO_LOCK_STATUS is derived from that enum.
    """
    lock = _make_lock()

    result = lock._parse_state(bytes.fromhex(frame_hex))

    assert result is not None
    assert list(result) == [expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0x09, LockStatus.UNLATCHING), (0x0A, LockStatus.UNLATCHED)],
    ids=["unlatching", "unlatched"],
)
def test_parse_lock_status_decodes_the_unlatch_states(
    value: int, expected: LockStatus, caplog: pytest.LogCaptureFixture
) -> None:
    """The second decode site takes the two values, and stops logging them.

    _parse_lock_status has four call sites: the DOOR_AND_LOCK branch, every
    lock_status() poll, the activity LOCK record's status byte, and the low
    nibble of the activity PIN record's status byte. It is therefore the site
    the enum change reaches beyond the pushed LOCK_ONLY frame above. Both
    values logged an "Unrecognized lock_status_str code" line at all four
    sites while the members were commented out; this change ends that
    diagnostic for them everywhere, including the two activity decodes.
    """
    lock = _make_lock()

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        assert lock._parse_lock_status(value) is expected

    assert "Unrecognized lock_status_str" not in caplog.text


def test_parse_success_op_response_with_0200_trailer_is_no_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The issue #317 fixture: byte[3] shifts with the plaintext trailer.

    A successful unlock op-response with the ``0200`` CommandType trailer
    moves byte[3] to 0x39. Keying off byte[3] would miss it; keying off
    byte[15]=0x00 recognizes it as a successful op-response with no state
    update -- and it must not log "Unknown state".
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a00390000000000000000000000000200")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def test_parse_lock_activity_is_no_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A LOCK_ACTIVITY (0xBB 0x2D) frame is recognized with no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb2d008000000000000000000000000000")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def test_mechanical_operation_errors_is_the_whole_mech_range() -> None:
    """The hand-written set holds exactly the MECH_* codes, and nothing else.

    A member dropped from it stops being a known mechanical failure: its log
    line is promoted to warning and reads as a result the decode has no story
    for. Derived here from the enum names so a drop fails rather than passes
    quietly.
    """
    assert {
        error for error in OperationError if error.name.startswith("MECH_")
    } == MECHANICAL_OPERATION_ERRORS


@pytest.mark.parametrize(
    ("awaited_opcode", "result_byte", "expected_level"),
    [
        *(
            (Commands.LOCK.value, error.value, "DEBUG")
            for error in sorted(MECHANICAL_OPERATION_ERRORS)
        ),
        (Commands.LOCK.value, 0x32, "WARNING"),
        (None, 0x1F, "WARNING"),
    ],
)
def test_parse_op_response_failure_log_level(
    awaited_opcode: int | None,
    result_byte: int,
    expected_level: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The mechanical result of an operation we issued logs at DEBUG, because
    OperationFailedError carries the same cause to the caller. A result we have
    no decode story for, and a mechanical result with no operation of ours
    awaiting that opcode (which came from the lock, the app or auto-lock and
    has no other record), both log at WARNING. All of them display JAMMED,
    since every failure class needs manual intervention at the lock.

    Every member of MECHANICAL_OPERATION_ERRORS is driven through, so dropping
    one from the set fails here rather than silently promoting its log line.
    """
    lock = _make_lock()
    lock._awaited_operation_opcode = awaited_opcode

    frame = bytearray.fromhex("bb0b001b00000000000000000000001f0000")
    frame[0x0F] = result_byte
    with caplog.at_level("DEBUG", logger="yalexs_ble.lock"):
        result = lock._parse_state(bytes(frame))

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    records = [
        record
        for record in caplog.records
        if "Operation failed with result" in record.message
    ]
    assert [record.levelname for record in records] == [expected_level]


def test_parse_non_mech_error_is_jammed_and_logs_decoded_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-MECH failure result still parses as JAMMED and logs its name."""
    lock = _make_lock()

    # byte[15] = 0x32 VBAT_LOW (synthetic; no real capture for a non-MECH error).
    # Captured at WARNING: an operation failure must be visible at default
    # log levels, not only in a debug session.
    frame = bytes.fromhex("bb0b00000000000000000000000000320000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert "0x32" in caplog.text
    assert "VBAT_LOW" in caplog.text


def test_parse_unknown_error_code_is_jammed_and_logs_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped non-zero result is JAMMED and logs the raw value as unknown."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0b00000000000000000000000000770000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert "0x77" in caplog.text
    assert "unknown" in caplog.text


def test_last_op_error_is_retained() -> None:
    """The op-response result byte[15] is retained on the lock instance."""
    # Collected and compared once: asserting on the attribute per step narrows
    # it (mypy keeps the narrowing across the _parse_state call) and the later
    # steps are then flagged unreachable.
    lock = _make_lock()
    seen: list[int | None] = [lock._last_op_error]

    lock._parse_state(bytes.fromhex("bb0b001b00000000000000000000001f0000"))
    seen.append(lock._last_op_error)

    lock._parse_state(bytes.fromhex("bb0b003a0000000000000000000000000000"))
    seen.append(lock._last_op_error)

    assert seen == [None, 0x1F, 0x00]


def test_parse_bogus_frame_is_none_and_logs_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame with an unrecognized flag byte is not recognized and still logs."""
    lock = _make_lock()

    frame = bytes.fromhex("cc00000000000000000000000000000000")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        assert lock._parse_state(frame) is None
        lock._internal_state_callback(frame)

    assert "Unknown state" in caplog.text


def test_internal_state_callback_emits_recognized_state() -> None:
    """A recognized frame with state content reaches the state callback."""
    received: list[list[LockStateValue]] = []
    lock = _make_lock(lambda states: received.append(list(states)))

    # Settled status push after a jam: GETSTATUS/LOCK_ONLY with state 0x07
    # (production capture).
    lock._internal_state_callback(bytes.fromhex("bb02003a0200000007000000000000000000"))

    assert received == [[LockStatus.JAMMED]]


def test_jammed_maps_to_the_settled_static_position_value() -> None:
    """JAMMED is the settled post-jam status value 0x07 (STATICPOSITION)."""
    assert LockStatus(0x07) is LockStatus.JAMMED
    assert VALUE_TO_LOCK_STATUS[0x07] is LockStatus.JAMMED


def _make_lock(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
    *,
    ack_callback: Callable[[], None] | None = None,
    op_response_callback: Callable[[], None] | None = None,
) -> Lock:
    return Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        state_callback,
        ack_callback=ack_callback,
        op_response_callback=op_response_callback,
    )


def test_parse_auto_lock_state_timed_from_wire() -> None:
    """Real capture: both uint16 timers set to 1800 -> Timed 30 min.

    Front Door READSETTING response, YUR/DEL fw 2.1.0 (2026-07-05 capture).
    """
    lock = _make_lock()
    response = bytes.fromhex("bb0400fb2800000008070807000000000000")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 1800)


def test_parse_auto_lock_state_off_from_wire() -> None:
    """Real capture: all-zero setting value -> auto-lock off.

    Back Door READSETTING response (2026-07-05 capture).
    """
    lock = _make_lock()
    response = bytes.fromhex("bb0400192800000000000000000000000000")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.OFF, 0)


def test_parse_auto_lock_state_old_encoding_reads_user_value() -> None:
    """A value written by a release before the two-timer encoding -> Timed 30.

    Earlier releases stored the user's seconds in the never-opened timer and a
    fixed 90 in the door-close timer, so Timed(30) was written as 1e 00 5a 00.
    The decode reports the never-opened timer, so the value reads back as set.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("1e005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 30)


def test_parse_auto_lock_state_zero_never_opened_falls_back() -> None:
    """A zero never-opened timer falls back to the door-close timer.

    Synthetic value exercising the branch; not a captured device value.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("00005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 90)


def test_parse_auto_lock_state_instant_never_opened_only() -> None:
    """Derivation branch: never-opened timer set, door-close timer zero -> Instant.

    Synthetic value exercising the branch; not a captured device value.
    """
    lock = _make_lock()
    response = bytes(8) + (0x0005).to_bytes(4, "little")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.INSTANT, 5)


class _CommandCaptureSession:
    """Minimal Session stand-in that captures executed commands.

    build_operation_command mirrors Session's 18-byte frame layout
    (EE, opcode, cmd byte at [4], ClearText trailer marker at [16]).
    """

    def __init__(self) -> None:
        self.sent: list[bytearray] = []

    def build_operation_command(self, opcode: int, cmd_byte: int) -> bytearray:
        cmd = bytearray(0x12)
        cmd[0x00] = 0xEE
        cmd[0x01] = opcode
        cmd[0x04] = cmd_byte
        cmd[0x10] = 0x02
        return cmd

    async def execute(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        self.sent.append(command)
        return b""


async def _set_auto_lock_payload(mode: AutoLockMode, duration: int) -> bytearray:
    """Run set_auto_lock against a capture session; return the sent command."""
    lock = _make_lock()
    session = _CommandCaptureSession()
    lock.session = session  # type: ignore[assignment]
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    await lock.set_auto_lock(mode, duration)
    assert len(session.sent) == 1
    return session.sent[0]


@pytest.mark.asyncio
async def test_set_auto_lock_timed_encodes_seconds_in_both_timers() -> None:
    """Timed(1800) -> both uint16 timers = 1800 -> [8:12] = 08 07 08 07."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 1800)
    assert cmd[0x01] == Commands.WRITESETTING.value
    assert cmd[0x04] == 0x28  # auto-lock setting id
    assert cmd[0x08:0x0C] == bytes.fromhex("08070807")


@pytest.mark.asyncio
async def test_set_auto_lock_instant_encodes_never_opened_only() -> None:
    """Instant(5) -> never-opened timer = 5, door-close 0 -> [8:12] = 05 00 00 00."""
    cmd = await _set_auto_lock_payload(AutoLockMode.INSTANT, 5)
    assert cmd[0x08:0x0C] == bytes.fromhex("05000000")


@pytest.mark.asyncio
async def test_set_auto_lock_off_encodes_zero() -> None:
    """Off -> value = 0 regardless of the duration argument."""
    cmd = await _set_auto_lock_payload(AutoLockMode.OFF, 1800)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_set_auto_lock_duration_out_of_range_raises() -> None:
    """Durations must fit a uint16 timer; 0xFFFF+ is rejected (app rule 1-65534)."""
    with pytest.raises(ValueError, match="out of range"):
        await _set_auto_lock_payload(AutoLockMode.TIMER, 0xFFFF)


@pytest.mark.asyncio
async def test_set_auto_lock_round_trips_through_decode() -> None:
    """A value we write, echoed back by the lock, decodes to what we set."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 1800)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(
        AutoLockMode.TIMER, 1800
    )


@pytest.mark.asyncio
async def test_set_auto_lock_timed_accepts_upper_bound() -> None:
    """Timed(0xFFFE) is the largest accepted duration.

    Both uint16 timers take the seconds, so [8:12] = fe ff fe ff.
    """
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0xFFFE)
    assert cmd[0x08:0x0C] == bytes.fromhex("fefffeff")


@pytest.mark.asyncio
async def test_set_auto_lock_timed_zero_duration_encodes_off_shape() -> None:
    """Timed with a zero duration collapses to the off shape: an all-zero value."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_auto_lock_status_issues_read() -> None:
    """auto_lock_status sends a READSETTING for the auto-lock setting.

    The wait completes on the acknowledgment, which carries no value, so the
    method returns nothing; the stored setting arrives later as a settings
    response on the notify path.
    """
    lock = _make_lock()
    session = _CommandCaptureSession()
    lock.session = session  # type: ignore[assignment]
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    await lock.auto_lock_status()
    assert len(session.sent) == 1
    assert session.sent[0][0x01] == Commands.READSETTING.value
    assert session.sent[0][0x04] == SettingType.AUTOLOCK.value


@pytest.mark.asyncio
async def test_set_auto_lock_instant_round_trips_through_decode() -> None:
    """Instant(5), encoded then decoded, returns Instant(5)."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.INSTANT, 5)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(AutoLockMode.INSTANT, 5)


@pytest.mark.asyncio
async def test_set_auto_lock_off_round_trips_through_decode() -> None:
    """Off, encoded then decoded, returns Off with a zero duration."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.OFF, 0)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(AutoLockMode.OFF, 0)


def test_parse_state_readsetting_ack_ignored() -> None:
    """The READSETTING (0x04) transport ACK carries no state -> recognized, ignored.

    Real ACK frame for an auto-lock READSETTING (2026-07-05 capture); must return
    an empty iterable (not None), so it is never logged as an unknown frame.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa0400282800000000000000000000000200")
    assert lock._parse_state(ack) == ()


def test_parse_state_writesetting_ack_ignored() -> None:
    """The WRITESETTING (0x03) transport ACK carries no state -> recognized, ignored.

    Real ACK frame for an auto-lock write of Timed(90) (2026-07-16 capture); the
    stored value is echoed at [8:12] but the frame is only the acknowledgment --
    the authoritative value is the 0xBB settings response that follows.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa030075280000005a005a00000000000200")
    assert lock._parse_state(ack) == ()


def test_parse_state_ack_for_other_opcode_is_unknown() -> None:
    """ACK recognition is scoped to the settings opcodes.

    An 0xAA frame whose opcode is neither a lock/unlock ack nor a settings ack
    is not claimed: it falls through to None, so a new acknowledgment type on
    another model still surfaces as an unknown frame instead of being silently
    dropped. Frame built from the READSETTING ACK capture above with the opcode
    byte changed to LOCK_ACTIVITY (0x2D), which is recognized on the 0xBB flag
    only.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa2d00282800000000000000000000000200")
    assert lock._parse_state(ack) is None


def test_settings_response_matcher_takes_value_frame_not_ack() -> None:
    """The matcher keys on 0xBB + the settings opcode + the setting id.

    All frames verbatim from the 2026-07-16 field capture: a settings command
    is answered by an 0xAA acknowledgment ~40 ms before the 0xBB value frame,
    and the acknowledgment's zero value field decodes as auto-lock off.
    """
    write_matcher = _settings_response_matcher(
        Commands.WRITESETTING.value, SettingType.AUTOLOCK.value
    )

    read_response = bytes.fromhex("bb0400fb2800000008070807000000000000")
    write_ack = bytes.fromhex("aa030075280000005a005a00000000000200")
    write_response = bytes.fromhex("bb030066280000005a005a00000000000000")
    battery_answer = bytes.fromhex("bb0200a50f00000079140000000000000200")

    assert write_matcher(write_response)
    assert not write_matcher(write_ack)
    assert not write_matcher(read_response)  # wrong opcode for the write
    assert not write_matcher(battery_answer)
    assert not write_matcher(write_response[:4])  # truncated below the setting id


_CHAR_DATA: dict[str, bytes] = {
    MODEL_NUMBER_CHARACTERISTIC: b"ASL-03",
    SERIAL_NUMBER_CHARACTERISTIC: b"12345",
    FIRMWARE_REVISION_CHARACTERISTIC: b"2.0.0",
}

# Model is read first, then serial, firmware.
_CHAR_ORDER: tuple[str, ...] = (
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    FIRMWARE_REVISION_CHARACTERISTIC,
)


def _make_lock_with_mock_client(
    side_effects: dict[str, Exception] | None = None,
) -> tuple[Lock, MagicMock]:
    """Create a Lock with a mock BLE client for lock_info tests."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", details=None),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )
    mock_client = MagicMock()
    mock_client.is_connected = True
    lock.client = mock_client
    lock.session = MagicMock()
    lock.secure_session = MagicMock()

    effects = side_effects or {}

    # Map each characteristic UUID to a unique mock object so
    # read_gatt_char can identify which UUID is being read.
    char_mocks: dict[str, MagicMock] = {}
    mock_to_uuid: dict[int, str] = {}
    for uuid in _CHAR_ORDER:
        m = MagicMock()
        char_mocks[uuid] = m
        mock_to_uuid[id(m)] = uuid

    mock_client.services.get_characteristic = char_mocks.get

    async def read_gatt_char(char: MagicMock) -> bytes:
        uuid = mock_to_uuid[id(char)]
        if uuid in effects:
            raise effects[uuid]
        return _CHAR_DATA[uuid]

    mock_client.read_gatt_char = read_gatt_char
    mock_client._mock_to_uuid = mock_to_uuid
    return lock, mock_client


@pytest.mark.asyncio
async def test_lock_info_success() -> None:
    """Test lock_info reads all characteristics successfully."""
    lock, _ = _make_lock_with_mock_client()

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )


@pytest.mark.asyncio
async def test_lock_info_partial_failure() -> None:
    """Test lock_info continues when individual reads fail."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={SERIAL_NUMBER_CHARACTERISTIC: BleakError("Connection dropped")}
    )

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_all_reads_fail() -> None:
    """Test lock_info returns all Unknown when every read fails."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={uuid: BleakError("Failed") for uuid in _CHAR_ORDER}
    )

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="",
        serial="aa:bb:cc:dd:ee:ff",
        firmware="Unknown",
    )


@pytest.mark.asyncio
async def test_lock_info_timeout() -> None:
    """Test lock_info returns partial results when reads hang."""
    lock, mock_client = _make_lock_with_mock_client()

    async def hang_forever(char: MagicMock) -> bytes:
        await asyncio.sleep(999)
        return b""  # unreachable

    mock_client.read_gatt_char = hang_forever

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0):
        info = await lock.lock_info()

    # All reads hung so no results, but we get defaults instead of an exception
    assert info.manufacturer == "Yale/August"
    assert info.model == ""
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "Unknown"


@pytest.mark.asyncio
async def test_lock_info_missing_characteristic() -> None:
    """Test lock_info skips missing characteristics instead of aborting."""
    lock, mock_client = _make_lock_with_mock_client()

    original_get = mock_client.services.get_characteristic

    def get_char_skip_serial(uuid: str) -> MagicMock | None:
        if uuid == SERIAL_NUMBER_CHARACTERISTIC:
            return None
        return original_get(uuid)

    mock_client.services.get_characteristic = get_char_skip_serial

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_reads_model_first() -> None:
    """Test that model is read first so it's available as early as possible."""
    lock, mock_client = _make_lock_with_mock_client()
    call_order: list[str] = []
    original_read = mock_client.read_gatt_char
    mock_to_uuid = mock_client._mock_to_uuid

    async def tracking_read(char: MagicMock) -> bytes:
        call_order.append(mock_to_uuid[id(char)])
        return await original_read(char)

    mock_client.read_gatt_char = tracking_read

    await lock.lock_info()

    assert call_order[0] == MODEL_NUMBER_CHARACTERISTIC


def test_ack_matcher_matches_only_the_written_operation() -> None:
    """The ack matcher keys on 0xAA + the written opcode + operation byte."""
    matches = _ack_matcher(0x0B, 0x04)

    # Correct ack: 0xAA, opcode 0x0B, operation byte 0x04.
    assert matches(bytes.fromhex("aa0b00450400000000000000000000000200"))
    # Same opcode but operation byte 0x00 -- a plain-lock ack, not securemode.
    assert not matches(bytes.fromhex("aa0b00490000000000000000000000000200"))
    # Wrong opcode (0x0A).
    assert not matches(bytes.fromhex("aa0a004a0000000000000000000000000200"))
    # An op-response (0xBB), not an acknowledgement.
    assert not matches(bytes.fromhex("bb0b00450400000000000000000000000200"))


def test_operation_response_matcher_matches_only_its_opcode() -> None:
    """The op-response matcher keys on 0xBB + the sent opcode, full length."""
    matches = _operation_response_matcher(0x0A)

    # An 18-byte 0xBB 0x0A op-response.
    assert matches(bytes.fromhex("bb0a00000000000000000000000000000200"))
    # Wrong opcode (0x0B).
    assert not matches(bytes.fromhex("bb0b00000000000000000000000000000200"))
    # An acknowledgement (0xAA), not an op-response.
    assert not matches(bytes.fromhex("aa0a00000000000000000000000000000200"))
    # Truncated: byte[15] (the result) is not present.
    assert not matches(bytes.fromhex("bb0a0000000000000000"))


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Yield to the event loop until predicate() holds (bounded)."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


def _make_connected_lock_with_session(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
    *,
    ack_callback: Callable[[], None] | None = None,
    op_response_callback: Callable[[], None] | None = None,
) -> Lock:
    """Build a connected Lock backed by a real Session over a mock BLE client.

    Mirrors tests/test_session.py: only cipher_encrypt is set, so notify frames
    pass through Session.decrypt unchanged and can be fed verbatim.
    """
    lock = _make_lock(
        state_callback,
        ack_callback=ack_callback,
        op_response_callback=op_response_callback,
    )
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    lock.client = client
    lock.secure_session = MagicMock()
    session = Session(
        client, "mylock", asyncio.Lock(), set(), lock._internal_state_callback
    )
    session.cipher_encrypt = Cipher(
        algorithms.AES(bytes(16)),
        modes.CBC(bytes(16)),
    ).encryptor()
    lock.session = session
    return lock


# --------------------------------------------------------------------------- #
# Mechanical operations through the staged session wait
# --------------------------------------------------------------------------- #


def _with_checksum(frame: bytearray) -> bytes:
    """Stamp byte[3] so the 18-byte simple checksum sums to zero (a valid frame).

    Synthetic frames, not field captures: built to the layout the matchers key
    on, for the cases no capture covers.
    """
    frame = bytearray(frame)
    frame[0x03] = 0
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


# Field-captured operation acknowledgements. The unlatch echo has no capture,
# so its frame is built by _ack_frame below.
_LOCK_ACK_HEX = "aa0b00490000000000000000000000000200"
_UNLOCK_ACK_HEX = "aa0a004a0000000000000000000000000200"
_SECUREMODE_ACK_HEX = "aa0b00450400000000000000000000000200"


def _ack_frame(opcode: int, operation_byte: int) -> bytes:
    """A 0xAA acknowledgement echoing an operation's opcode and operation byte."""
    frame = bytearray(0x12)
    frame[0x00] = 0xAA
    frame[0x01] = opcode
    frame[0x04] = operation_byte
    return _with_checksum(frame)


def _op_response_frame(opcode: int, result: int = OperationError.COMM_SUCCESS) -> bytes:
    """A 0xBB op-response carrying the operation result in byte[15]."""
    frame = bytearray(0x12)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x0F] = result
    return _with_checksum(frame)


async def _drive_operation(lock: Lock, op_attr: str, opcode: int, ack: bytes) -> None:
    """Run a force_* method, feeding its ack then op-response through notify.

    The acknowledgement has to be matched before the op-response is fed. A
    command carrying the wrong operation byte, or a matcher that never
    matches, would otherwise still complete on the op-response alone and the
    operation would look correct.
    """
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(ack))
        assert session._ack_future is None, "the acknowledgement was not matched"
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(opcode)))

    feeder = asyncio.create_task(feed())
    await getattr(lock, op_attr)()
    await feeder


def test_parse_operation_ack_reports_no_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operation acks (0xAA LOCK/UNLOCK) are recognized but carry no state.

    They echo the request type with no result, so a securemode request (the
    0x0B Lock echo) used to display a false LOCKED. State now comes from the
    op-response; the ack is recognized (empty iterable), emits nothing, and
    must not surface as an unknown frame.
    """
    states: list[list[LockStateValue]] = []
    lock = _make_lock(lambda s: states.append(list(s)))

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        for frame_hex in (
            "aa0b00490000000000000000000000000200",
            "aa0a004a0000000000000000000000000200",
        ):
            frame = bytes.fromhex(frame_hex)
            result = lock._parse_state(frame)
            assert result is not None
            assert list(result) == []
            lock._internal_state_callback(frame)

    assert states == []  # the state callback was never invoked
    assert "Unknown state" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op_attr", "opcode", "ack"),
    [
        ("force_lock", Commands.LOCK, bytes.fromhex(_LOCK_ACK_HEX)),
        ("force_unlock", Commands.UNLOCK, bytes.fromhex(_UNLOCK_ACK_HEX)),
        ("force_securemode", Commands.LOCK, bytes.fromhex(_SECUREMODE_ACK_HEX)),
        (
            "force_unlatch",
            Commands.UNLOCK,
            _ack_frame(Commands.UNLOCK, UNLATCH_OPERATION_BYTE),
        ),
    ],
    ids=["lock", "unlock", "securemode", "unlatch"],
)
async def test_force_operations_complete_on_ack_then_op_response(
    op_attr: str, opcode: int, ack: bytes
) -> None:
    """Each force_* completes only on its own ack, then its 0xBB op-response.

    Drives _execute_operation_command end to end through the real staged
    session wait. The lock, unlock and securemode acknowledgements are field
    captures (the securemode one appears 179 times across two locks in the
    logs); the unlatch one has no capture and is built to the same layout.
    byte[4] is the operation byte the command must have carried, so the
    acknowledgement only matches if the right command went out.
    byte[15]=COMM_SUCCESS in the op-response makes the method return rather
    than raise OperationFailedError.
    """
    lock = _make_connected_lock_with_session()

    await _drive_operation(lock, op_attr, opcode, ack)

    # The operation ran to its op-response and reported success, so nothing was
    # left awaited on this instance.
    assert lock._awaited_operation_opcode is None


@pytest.mark.asyncio
async def test_an_op_response_for_another_opcode_does_not_complete_the_wait() -> None:
    """The staged wait completes only on the op-response echoing its opcode.

    While a force_lock is in flight, an op-response carrying the Unlock opcode
    lands first. It must leave the wait armed; only the op-response echoing
    the Lock opcode completes the operation, so the result is read from the
    right frame.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(
            0, bytearray(bytes.fromhex("aa0b00490000000000000000000000000200"))
        )
        assert session._ack_future is None, "the acknowledgement was not matched"
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.UNLOCK)))
        assert session._notify_future is not None, (
            "an op-response for another opcode completed the wait"
        )
        session._notify(0, bytearray(_op_response_frame(Commands.LOCK)))

    feeder = asyncio.create_task(feed())
    await lock.force_lock()
    await feeder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapper", "force_attr"),
    [
        ("securemode", "force_securemode"),
        ("lock", "force_lock"),
        ("unlock", "force_unlock"),
    ],
    ids=["securemode", "lock", "unlock"],
)
async def test_convenience_wrappers_run_the_operation_outside_the_target_state(
    wrapper: str, force_attr: str
) -> None:
    """A wrapper finding the lock outside its target state runs the operation.

    The wrappers return None either way; a reported failure surfaces as the
    OperationFailedError the force_* call raises, so the delegation is the
    whole contract.
    """
    lock = _make_lock()

    with (
        patch.object(lock, "lock_status", AsyncMock(return_value=LockStatus.UNKNOWN)),
        patch.object(lock, force_attr, AsyncMock()) as mock_force,
    ):
        await getattr(lock, wrapper)()

    mock_force.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapper", "target_status", "force_attr"),
    [
        ("securemode", LockStatus.SECUREMODE, "force_securemode"),
        ("lock", LockStatus.LOCKED, "force_lock"),
        ("unlock", LockStatus.UNLOCKED, "force_unlock"),
    ],
    ids=["securemode", "lock", "unlock"],
)
async def test_convenience_wrappers_skip_the_operation_in_the_target_state(
    wrapper: str, target_status: LockStatus, force_attr: str
) -> None:
    """A wrapper finding the lock already in its target state issues nothing.

    No operation is issued, so nothing could have failed and the caller's
    goal state holds.
    """
    lock = _make_lock()

    with (
        patch.object(lock, "lock_status", AsyncMock(return_value=target_status)),
        patch.object(lock, force_attr, AsyncMock()) as mock_force,
    ):
        await getattr(lock, wrapper)()

    mock_force.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_unlatch_wrapper_runs_force_unlatch() -> None:
    """Lock.unlatch() always fires force_unlatch and completes on its op-response.

    Unlatch is momentary, so there is no steady "unlatched" state to
    short-circuit on the way lock()/unlock() do; the wrapper returns once the
    op-response arrives.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(
            0, bytearray(_ack_frame(Commands.UNLOCK, UNLATCH_OPERATION_BYTE))
        )
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.UNLOCK)))

    feeder = asyncio.create_task(feed())
    # The wrapper returns nothing, as its three siblings now do; a reported
    # failure would surface as OperationFailedError.
    await lock.unlatch()
    await feeder


@pytest.mark.asyncio
async def test_force_unlatch_grants_the_extended_op_response_budget() -> None:
    """Unlatch gets the longer op-response budget because it is a longer motion.

    Unlatch pulls the latch in (~2 s), holds it there (~6 s) and releases it
    (~2 s), so its op-response -- emitted when the motor stops -- lands ~10 s
    after the write, against ~3 s for a plain lock or unlock. force_unlatch
    passes UNLATCH_OPERATION_RESPONSE_TIMEOUT, and encodes unlatch as the
    Unlock opcode with the unlatch operation byte (there is no dedicated
    unlatch opcode).

    The budget and the byte are asserted against the wire values, not against
    the constants that produced them: byte 0x01 is what #351 sent, and a
    budget equal to the plain one is the extension reversed.
    """
    lock = _make_lock()
    session = MagicMock()

    def _build(opcode: int, cmd_byte: int) -> bytearray:
        cmd = bytearray(0x12)
        cmd[0x01] = opcode
        cmd[0x04] = cmd_byte
        return cmd

    session.build_operation_command.side_effect = _build
    lock.session = session
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    captured_command: bytearray | None = None
    captured_timeout: float | None = None

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        nonlocal captured_command, captured_timeout
        captured_command = command
        captured_timeout = response_timeout

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    await lock.force_unlatch()
    assert captured_timeout == UNLATCH_OPERATION_RESPONSE_TIMEOUT
    assert UNLATCH_OPERATION_RESPONSE_TIMEOUT > OPERATION_RESPONSE_TIMEOUT
    assert captured_command is not None
    assert captured_command[0x01] == 0x0A  # the Unlock opcode
    assert captured_command[0x04] == 0x0A  # the unlatch operation byte


@pytest.mark.asyncio
async def test_every_operation_names_the_budget_its_motion_needs() -> None:
    """Each operation passes its own response budget; none is defaulted.

    The budget is a property of the motion, so it is set where the operation
    is issued rather than inherited from a signature: lock, unlock and
    securemode drive a bolt, an unlatch also pulls the latch in and holds it
    out. _execute_operation_command takes it as a required argument, which is
    what makes an operation added later choose rather than inherit.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    budgets: dict[str, float] = {}

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        budgets[command_name] = response_timeout

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    await lock.force_lock()
    await lock.force_unlock()
    await lock.force_securemode()
    await lock.force_unlatch()

    assert budgets == {
        "force_lock": OPERATION_RESPONSE_TIMEOUT,
        "force_unlock": OPERATION_RESPONSE_TIMEOUT,
        "force_securemode": OPERATION_RESPONSE_TIMEOUT,
        "force_unlatch": UNLATCH_OPERATION_RESPONSE_TIMEOUT,
    }


@pytest.mark.asyncio
async def test_unlatch_is_the_only_operation_that_skips_the_ack_wait() -> None:
    """force_unlatch waits on its op-response alone; the other three keep the
    acknowledgement stage.

    The acknowledgement is an early delivery signal and it pays only where the
    caller may re-send. Lock, unlock and securemode may, so a missing
    acknowledgement should end their attempt early. An unlatch may not once its
    command is written, and its op-response lands well past the acknowledgement
    budget, so the stage could only end the operation while the latch was out.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    waited: dict[str, bool] = {}

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        waited[command_name] = wait_for_ack

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    await lock.force_lock()
    await lock.force_unlock()
    await lock.force_securemode()
    await lock.force_unlatch()

    assert waited == {
        "force_lock": True,
        "force_unlock": True,
        "force_securemode": True,
        "force_unlatch": False,
    }


@pytest.mark.asyncio
async def test_force_unlatch_failure_before_write_stays_retryable() -> None:
    """A failure before the command write leaves write_attempted False, so the
    error propagates unchanged and the caller may retry -- the latch never fired.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        raise DisconnectedError("dropped before the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(DisconnectedError):
        await lock.force_unlatch()


@pytest.mark.asyncio
async def test_force_unlatch_failure_after_write_converts_to_unlatch_error() -> None:
    """A retryable failure AFTER the command write converts to the non-retryable
    UnlatchError: a repeated unlatch fires the latch again, so it must not retry.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        assert progress is not None
        progress.write_attempted = True
        raise TimeoutError("no op-response after the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    # The originating error is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_force_unlatch_errored_write_converts_to_unlatch_error() -> None:
    """A write call that errors leaves delivery unknown: the request PDU may
    have reached the lock even though the write reported failure, so the
    failure converts to the non-retryable UnlatchError instead of retrying.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        assert progress is not None
        progress.write_attempted = True
        raise BleakError("link dropped during the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    # The originating error is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, BleakError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        EOFError("dbus socket closed"),
        BrokenPipeError("dbus socket closed"),
        AttributeError("backend went away mid-write"),
        ValueError("an error from nowhere near the retry set"),
    ],
)
async def test_force_unlatch_converts_every_post_write_failure(
    error: Exception,
) -> None:
    """The post-write conversion is type-blind, so no failure can re-send.

    Enumerating retryable types is what let this leak: the retry set is
    assembled from bleak_retry_connector and grows without reference to this
    guard. The first three of these were in that set and outside an earlier
    enumeration, so a post-write escape was retried and the latch fired
    twice; the fourth is in no retry set at all and converts just the same,
    which is the property that keeps the invariant true as the set widens.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        assert progress is not None
        progress.write_attempted = True
        raise error

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    assert excinfo.value.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        OperationIncompleteError("acked but no op-response"),
        OperationFailedError("op failed", 0x1F),
    ],
    ids=["incomplete", "failed"],
)
async def test_force_unlatch_operation_result_errors_are_not_converted(
    error: Exception,
) -> None:
    """Both operation-result types are already non-retryable, so each
    propagates as itself even after the write; neither is re-wrapped as
    UnlatchError. The identity assertion is what pins the two-member
    passthrough tuple: wrapping either type would break it.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float,
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
    ) -> None:
        assert progress is not None
        progress.write_attempted = True
        raise error

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(type(error)) as excinfo:
        await lock.force_unlatch()
    assert excinfo.value is error


def test_operation_failed_error_survives_being_copied() -> None:
    """The one exception here carrying data must rebuild from its own args.

    BaseException reconstructs from self.args, which holds the message alone,
    so a copy of this type would call __init__ an argument short and raise a
    TypeError that hides the failure it was reporting. Copy and pickle both
    rebuild through __reduce__, so copying exercises the path either takes.
    Its two siblings take no extra argument and cannot show the defect.
    """
    error = OperationFailedError("force_lock reported failure 0x1F", 0x1F)

    for rebuilt in (copy.copy(error), copy.deepcopy(error)):
        assert isinstance(rebuilt, OperationFailedError)
        assert rebuilt.result == 0x1F
        assert str(rebuilt) == str(error)


def test_unlatch_error_is_reachable_from_the_package_root() -> None:
    """The type a caller has to catch is importable where callers import from.

    Every test here takes it from yalexs_ble.session, so nothing else in the
    suite would notice if the package-root export were dropped, and a consumer
    catching it by the documented path would stop compiling.
    """
    assert yalexs_ble.UnlatchError is UnlatchError
    assert "UnlatchError" in yalexs_ble.__all__


@pytest.mark.asyncio
async def test_force_lock_failure_op_response_raises_operation_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure op-response (byte[15] != 0) raises OperationFailedError.

    The exchange completed and the lock named the cause, so the exception
    carries the result byte and is not converted or retried. The frame is
    built here, with byte[15] = 0x1F MECH_POSITION, the result a jammed lock
    reports. The failure record logs at DEBUG, which joins the awaited-opcode
    arming to its consumer: the parser saw the opcode armed when the frame
    landed, and an unarmed field routes the same record to WARNING.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(_ack_frame(Commands.LOCK, 0x00)))
        await asyncio.sleep(0)
        session._notify(
            0,
            bytearray(_op_response_frame(Commands.LOCK, OperationError.MECH_POSITION)),
        )

    feeder = asyncio.create_task(feed())
    with (
        caplog.at_level("DEBUG", logger="yalexs_ble.lock"),
        pytest.raises(OperationFailedError) as excinfo,
    ):
        await lock.force_lock()
    await feeder
    assert excinfo.value.result == OperationError.MECH_POSITION
    records = [
        record
        for record in caplog.records
        if "Operation failed with result" in record.message
    ]
    assert [record.levelname for record in records] == ["DEBUG"]
    # Cleared on the way out of the failure too: left set, every later external
    # op-response on that opcode would read as one of ours and log at debug for
    # the instance's life.
    assert lock._awaited_operation_opcode is None


@pytest.mark.asyncio
async def test_the_awaited_opcode_is_armed_at_the_command_write() -> None:
    """Nothing is awaited until the command has actually been written.

    The cooldown wait, the session lock and the write itself sit between the
    call and the command leaving the radio, and both wait futures are armed
    ahead of the write. An op-response landing in that stretch is not ours:
    no wait of ours could have taken it either. The field is armed from the
    write-success hook, the first moment one of our own op-responses can
    exist.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None
    at_write: list[int | None] = []
    at_hook: list[int | None] = []

    async def _write(*args: object, **kwargs: object) -> None:
        at_write.append(lock._awaited_operation_opcode)

    def _on_write_success() -> None:
        at_hook.append(lock._awaited_operation_opcode)

    assert lock.client is not None
    lock.client.write_gatt_char = AsyncMock(side_effect=_write)

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(bytes.fromhex(_LOCK_ACK_HEX)))
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.LOCK)))

    feeder = asyncio.create_task(feed())
    await lock.force_lock(write_success_callback=_on_write_success)
    await feeder

    # Nothing awaited while the command was being written, and armed by the
    # time the caller's own write-success hook ran.
    assert at_write == [None]
    assert at_hook == [Commands.LOCK.value]
    assert lock._awaited_operation_opcode is None


def test_ack_and_op_response_callbacks_fire() -> None:
    """ack_callback fires per operation ack; op_response_callback per op-response.

    The op-response hook fires for both a success and a failure result,
    independent of whatever state the parse emits.
    """
    ack_calls: list[int] = []
    op_calls: list[int] = []
    lock = _make_lock(
        ack_callback=lambda: ack_calls.append(1),
        op_response_callback=lambda: op_calls.append(1),
    )

    # Each operation ack fires ack_callback exactly once and emits no state.
    lock_ack = lock._parse_state(bytes.fromhex("aa0b00490000000000000000000000000200"))
    unlock_ack = lock._parse_state(
        bytes.fromhex("aa0a004a0000000000000000000000000200")
    )

    assert lock_ack is not None
    assert unlock_ack is not None
    assert list(lock_ack) == []
    assert list(unlock_ack) == []
    assert ack_calls == [1, 1]
    assert op_calls == []  # acks are not op-responses

    # A success op-response (byte[15]=0x00) and a failure one (byte[15]=0x1F)
    # both stamp op_response_callback, regardless of the emitted state.
    success = lock._parse_state(bytes.fromhex("bb0a003b0000000000000000000000000000"))
    failure = lock._parse_state(bytes.fromhex("bb0a001c00000000000000000000001f0000"))

    assert op_calls == [1, 1]
    assert success is not None
    assert failure is not None
    assert list(success) == []
    assert list(failure) == [LockStatus.JAMMED]
    # The op-responses did not spuriously fire the ack hook.
    assert ack_calls == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["ack_callback", "op_response_callback"])
async def test_a_raising_stream_hook_does_not_abort_the_operation(
    hook: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A hook that raises is contained and the operation still completes.

    Both hooks run while the frame is being parsed, and the session parses a
    frame before it resolves either wait future, so an exception escaping one
    would cost an operation that succeeded its op-response: the wait would run
    out its budget and report the result as never delivered.
    """
    calls: list[str] = []

    def _boom() -> None:
        calls.append(hook)
        raise RuntimeError("hook bug")

    lock = _make_connected_lock_with_session(
        ack_callback=_boom if hook == "ack_callback" else None,
        op_response_callback=_boom if hook == "op_response_callback" else None,
    )
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(bytes.fromhex(_LOCK_ACK_HEX)))
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.LOCK)))

    feeder = asyncio.create_task(feed())
    with caplog.at_level("ERROR", logger="yalexs_ble.lock"):
        await lock.force_lock()
    await feeder

    assert calls == [hook]
    assert f"{hook} raised, continuing to parse the frame" in caplog.text
    assert "hook bug" in caplog.text
