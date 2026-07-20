import asyncio
import contextlib
from collections.abc import Callable, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from bleak_retry_connector import BLEDevice

from yalexs_ble.const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
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
    SettingType,
)
from yalexs_ble.lock import (
    AA_BATTERY_VOLTAGE_TO_PERCENTAGE,
    Lock,
    _settings_response_matcher,
    convert_voltage_to_percentage,
)


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


def test_parse_ack_still_reports_state() -> None:
    """The AA transport-ack path is unchanged by the op-response decode."""
    lock = _make_lock()

    result = lock._parse_state(bytes.fromhex("aa0b00490000000000000000000000000200"))
    assert result is not None
    assert list(result) == [LockStatus.LOCKED]


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
) -> Lock:
    return Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        state_callback,
    )


def test_parse_auto_lock_state_timed_from_wire() -> None:
    """Real capture: value 1800 stored as seconds|(seconds<<16) -> Timed 30 min.

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

    Earlier releases stored the user's seconds in the never-opened half and a
    fixed 90 in the door-close half, so Timed(30) was written as 1e 00 5a 00.
    The decode reports the never-opened half, so the value reads back as set.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("1e005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 30)


def test_parse_auto_lock_state_zero_never_opened_falls_back() -> None:
    """A zero never-opened half falls back to the door-close half.

    Synthetic value exercising the branch; not a captured device value.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("00005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 90)


def test_parse_auto_lock_state_instant_low_half_only() -> None:
    """Derivation branch: low 16 bits set, high half zero -> Instant.

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
async def test_set_auto_lock_timed_encodes_seconds_in_both_halves() -> None:
    """Timed(1800) -> value = 1800|(1800<<16) -> [8:12] = 08 07 08 07."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 1800)
    assert cmd[0x01] == Commands.WRITESETTING.value
    assert cmd[0x04] == 0x28  # auto-lock setting id
    assert cmd[0x08:0x0C] == bytes.fromhex("08070807")


@pytest.mark.asyncio
async def test_set_auto_lock_instant_encodes_low_half_only() -> None:
    """Instant(5) -> value = 5 -> [8:12] = 05 00 00 00."""
    cmd = await _set_auto_lock_payload(AutoLockMode.INSTANT, 5)
    assert cmd[0x08:0x0C] == bytes.fromhex("05000000")


@pytest.mark.asyncio
async def test_set_auto_lock_off_encodes_zero() -> None:
    """Off -> value = 0 regardless of the duration argument."""
    cmd = await _set_auto_lock_payload(AutoLockMode.OFF, 1800)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_set_auto_lock_duration_out_of_range_raises() -> None:
    """Durations must fit a 16-bit half; 0xFFFF+ is rejected (app rule 1-65534)."""
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

    Both 16-bit halves take the seconds, so [8:12] = 0xFFFE|(0xFFFE<<16).
    """
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0xFFFE)
    assert cmd[0x08:0x0C] == bytes.fromhex("fefffeff")


@pytest.mark.asyncio
async def test_set_auto_lock_timed_zero_duration_encodes_off_shape() -> None:
    """Timed with a zero duration collapses to the off shape: an all-zero value."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_auto_lock_status_issues_read_and_returns_none() -> None:
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
    assert await lock.auto_lock_status() is None
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
