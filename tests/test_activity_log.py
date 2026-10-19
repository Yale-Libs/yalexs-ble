from collections.abc import Callable
from enum import IntEnum

import pytest

import yalexs_ble
from yalexs_ble.activity_log import (
    AllCodesDeleted,
    BatteryInfo,
    BCMTest,
    CodeAdded,
    CodeAddedv2,
    CodeDeleted,
    CodeDeletedv2,
    CodeOperation,
    CredentialAdded,
    CredentialDeleted,
    CredentialOperation,
    DialogReboot,
    DoorStateChanged,
    DPSInfo,
    FactoryReset,
    GetLogError,
    HomekeyOperation,
    KeypadClose,
    KeypadInfo,
    KeypadOpen,
    KeypadWrongCode,
    LocalEventNotification,
    LockOperation,
    LogRecord,
    MatterMessage,
    ProductMessage,
    Tamper,
    parse_log_event_packet,
)
from yalexs_ble.const import (
    CREDENTIAL_TYPE_NAMES_YALE,
    DOOR_STATE_NAMES,
    LOCK_STATE_NAMES_YALE,
    LOG_TYPE_GET_LOG_ERROR,
    LOG_TYPE_NAMES_YALE,
    OP_CODE_NAMES_YALE,
    RESPONSE_FRAME_LEN,
    TAMPER_TYPE_NAMES_YALE,
    DoorStatus,
    LockCred,
    LockStatus,
    LogType,
    OpCode,
    TamperType,
)
from yalexs_ble.util import _simple_checksum

# Captured GetLog replies.
KEYPAD_UNLOCK_SLOT_205 = bytes.fromhex("bb2d0072079aa176a333004bea14cd000200")
KEYPAD_UNLOCK_SLOT_200 = bytes.fromhex("bb2d00a4077a9e76a333004cdf14c8000200")
KEYPAD_UNLOCK_SLOT_UNKNOWN = bytes.fromhex("bb2d00d007299c76a3330049e414eeff0200")
KEYPAD_LOCK_BUTTON = bytes.fromhex("bb2d0056000b0500a0a176a3df14194a0200")
END_OF_LOG = bytes.fromhex("bb2d00de800200009aa176a3a8a176a30200")

# Captured GetLog replies from a Yale Unity L9000ER on 2026-10-25.
AUTO_RELOCK_LOCKED = bytes.fromhex("bb2d000e000505007baddd6a2017193f0200")
DOOR_STATE_CHANGED_SLOT_0 = bytes.fromhex("bb2d00080000030093aedd6a1f1719340200")
CREDENTIAL_OPERATION_RF_SLOT_7 = bytes.fromhex("bb2d00300e51addd6a01070013421f170200")
DOOR_CLOSED = bytes.fromhex("bb2d00a4205daddd6a010000000000000200")
DOOR_OPENED = bytes.fromhex("bb2d00ab2054addd6a030000000000000200")
# Captured from a Yale Unity L9000K3 on 2026-10-25.
INSIDE_BUTTON_LOCKED = bytes.fromhex("bb2d00d4000c0500a7eedd6a1914190f0200")
# Captured GetLog replies from a Yale Unity L9000ER on 2026-10-27.
DOOR_STATE_CHANGED_SLOT_255 = bytes.fromhex("bb2d001c000005ff0433e06a0817193d0200")
KEYPAD_INFO = bytes.fromhex("bb2d00c404922fe06ac30000008000000200")
CODE_OPERATION_SLOT_1 = bytes.fromhex("bb2d001d07fc2fe06a1300470b1701000200")

# The time stamp written into built frames, as hex in the frame and as a number.
TIME = "78563412"
TIME_STAMP = 0x12345678


def _built(log_type: int, fields: dict[int, str] | None = None) -> bytes:
    """Build a reply from the record layout; no lock sent a built frame.

    fields maps a byte offset to the hex written from that offset on.
    """
    frame = bytearray(RESPONSE_FRAME_LEN)
    frame[0] = 0xBB
    frame[1] = 0x2D
    frame[4] = log_type
    for offset, value in (fields or {}).items():
        data = bytes.fromhex(value)
        frame[offset : offset + len(data)] = data
    frame[3] = _simple_checksum(frame)
    return bytes(frame)


def _case(
    record: Callable[..., LogRecord],
    log_type: int,
    fields: dict[int, str],
    **values: object,
) -> tuple[bytes, LogRecord]:
    """A built frame and the record its bytes decode to."""
    frame = _built(log_type, fields)
    return frame, record(raw=frame, log_type=log_type, **values)


def _captured_case(
    record: Callable[..., LogRecord], frame: bytes, **values: object
) -> tuple[bytes, LogRecord]:
    """A captured frame and the record its bytes decode to."""
    return frame, record(raw=frame, log_type=frame[4], **values)


def _case_id(case: tuple[bytes, LogRecord]) -> str:
    return type(case[1]).__name__


def _assert_record(record: LogRecord, expected: LogRecord) -> None:
    """Compare two records; the repr tells an enumeration member from its integer."""
    assert record == expected
    assert repr(record) == repr(expected)


@pytest.mark.parametrize(
    ("frame", "time_stamp", "battery_level_percent", "battery_level", "slot"),
    [
        (KEYPAD_UNLOCK_SLOT_205, 0xA376A19A, 75, 5354, 205),
        (KEYPAD_UNLOCK_SLOT_200, 0xA3769E7A, 76, 5343, 200),
        (KEYPAD_UNLOCK_SLOT_UNKNOWN, 0xA3769C29, 73, 5348, 65518),
    ],
)
def test_captured_code_operation_records(
    frame: bytes,
    time_stamp: int,
    battery_level_percent: int,
    battery_level: int,
    slot: int,
) -> None:
    """Captured frames: the three CodeOperation records."""
    _assert_record(
        parse_log_event_packet(frame),
        CodeOperation(
            raw=frame,
            log_type=0x07,
            time_stamp=time_stamp,
            current_lock_state=LockStatus.UNLOCKED,
            current_door_state=DoorStatus.OPENED,
            slot=slot,
            error=0,
            battery_level=battery_level,
            battery_level_percent=battery_level_percent,
        ),
    )


def test_captured_lock_operation_record() -> None:
    """Captured frame: byte 15 is given as both error and battery_level_percent."""
    _assert_record(
        parse_log_event_packet(KEYPAD_LOCK_BUTTON),
        LockOperation(
            raw=KEYPAD_LOCK_BUTTON,
            log_type=0x00,
            op_code=OpCode.ONE_TOUCH,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0xA376A1A0,
            battery_level=5343,
            temperature=25,
            error=74,
            battery_level_percent=74,
        ),
    )


def test_captured_get_log_error_record() -> None:
    """Captured frame: the record an empty log answers with."""
    _assert_record(
        parse_log_event_packet(END_OF_LOG),
        GetLogError(
            raw=END_OF_LOG,
            log_type=0x80,
            error=2,
            key_slot=0,
            log_count=0,
            dirty_bit=0,
            dirty_bit_set_time=0xA376A19A,
            dirty_bit_cleared_time=0xA376A1A8,
        ),
    )


def test_code_operation_reads_both_states_and_the_error() -> None:
    """Built frame: byte 9 holds the lock state low and the door state high."""
    frame = _built(0x07, {5: TIME, 9: "15", 10: "05"})
    record = parse_log_event_packet(frame)
    assert isinstance(record, CodeOperation)
    assert record.current_lock_state is LockStatus.LOCKED
    assert record.current_door_state is DoorStatus.CLOSED
    assert record.error == 5


@pytest.mark.parametrize(
    ("log_type", "record_class"),
    [
        (0x0C, CredentialAdded),
        (0x0D, CredentialDeleted),
        (0x0E, CredentialOperation),
    ],
    ids=["CredentialAdded", "CredentialDeleted", "CredentialOperation"],
)
def test_credential_records_with_states(
    log_type: int, record_class: Callable[..., LogRecord]
) -> None:
    """Built frames: byte 12 holds the two states and the slot is two bytes."""
    frame = _built(
        log_type, {5: TIME, 9: "01", 10: "eeff", 12: "13", 13: "4b", 14: "ea14"}
    )
    _assert_record(
        parse_log_event_packet(frame),
        record_class(
            raw=frame,
            log_type=log_type,
            time_stamp=TIME_STAMP,
            credential_type=LockCred.RF,
            slot=65518,
            credential_error=None,
            current_lock_state=LockStatus.UNLOCKED,
            current_door_state=DoorStatus.CLOSED,
            battery_level=5354,
            battery_level_percent=75,
        ),
    )


def test_credential_operation_with_an_error() -> None:
    """Built frame: a high nibble of 0xF makes byte 12 the credential error."""
    record = parse_log_event_packet(_built(0x0E, {12: "f3"}))
    assert isinstance(record, CredentialOperation)
    assert record.credential_error == 0xF3
    assert record.current_lock_state is None
    assert record.current_door_state is None


def test_credential_type_without_a_member_stays_an_integer() -> None:
    """Built frame: credential type 3 has no name."""
    record = parse_log_event_packet(_built(0x0E, {9: "03"}))
    assert isinstance(record, CredentialOperation)
    assert record.credential_type == 3
    assert type(record.credential_type) is int


def test_lock_operation_with_an_unnamed_op_code_is_returned() -> None:
    """Built frame: opCode 0x0D has no member and stays an integer."""
    record = parse_log_event_packet(_built(0x00, {5: "0d", 6: "05"}))
    assert isinstance(record, LockOperation)
    assert record.op_code == 13
    assert type(record.op_code) is int


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0x0B, LockStatus.INTEGRATED_OTHER_SYSTEM),
        (0x0D, LockStatus.MASTER_CODE_NOT_SET),
        (0x0E, LockStatus.DEFAULT_MASTER_CODE_SET),
        (0x0F, LockStatus.LOCK_NOT_HANDED),
        (0x10, LockStatus.SHABBAT_MODE),
        (0x11, LockStatus.INTERNAL_FORCED_LOCKED),
        (0x08, LockStatus.UNKNOWN),
    ],
    ids=["0x0b", "0x0d", "0x0e", "0x0f", "0x10", "0x11", "0x08"],
)
def test_lock_operation_lock_state_decodes_to_its_lock_status(
    value: int, expected: LockStatus
) -> None:
    """Built frames: a named lock state is its member, and 0x08 reads UNKNOWN."""
    record = parse_log_event_packet(_built(0x00, {5: "0b", 6: f"{value:02x}"}))
    assert isinstance(record, LockOperation)
    assert record.current_lock_state is expected


def test_get_log_error_bit_fields() -> None:
    """Built frame: log_count crosses the byte boundary of the two bytes at 5."""
    record = parse_log_event_packet(_built(0x80, {5: "1a2b"}))
    assert isinstance(record, GetLogError)
    assert record.error == 2
    assert record.dirty_bit == 1
    assert record.log_count == 0x2B1


@pytest.mark.parametrize("log_type", [0x00, 0x80], ids=["LockOperation", "GetLogError"])
def test_key_slot_is_byte_7(log_type: int) -> None:
    """Built frames: byte 7 is key_slot in a LockOperation and a GetLogError record."""
    record = parse_log_event_packet(_built(log_type, {7: "07"}))
    assert isinstance(record, (LockOperation, GetLogError))
    assert record.key_slot == 7


def test_keypad_open_keycode() -> None:
    """Built frame: the seven code bytes print as 16 hex digits."""
    frame = _built(0x01, {5: TIME, 9: "123456ffffffff"})
    _assert_record(
        parse_log_event_packet(frame),
        KeypadOpen(
            raw=frame, log_type=0x01, time_stamp=TIME_STAMP, keycode="123456FFFFFFFFFF"
        ),
    )


def test_code_added_slot_and_code() -> None:
    """Built frame: the six code bytes are padded to 16 hex digits."""
    frame = _built(0x03, {5: TIME, 9: "07", 10: "1234ffffffff"})
    _assert_record(
        parse_log_event_packet(frame),
        CodeAdded(
            raw=frame,
            log_type=0x03,
            time_stamp=TIME_STAMP,
            slot=7,
            code="1234FFFFFFFFFFFF",
        ),
    )


def test_code_added_v2_code_in_hex() -> None:
    """Built frame: the five code bytes are one little-endian number."""
    frame = _built(0x0A, {5: TIME, 9: "0102", 11: "40e2010000"})
    _assert_record(
        parse_log_event_packet(frame),
        CodeAddedv2(
            raw=frame,
            log_type=0x0A,
            time_stamp=TIME_STAMP,
            slot=0x0201,
            code="000000000001E240",
        ),
    )


def test_battery_info_temperature_is_signed() -> None:
    """Built frame: byte 11 is a signed number."""
    frame = _built(0x15, {5: TIME, 9: "ea14", 11: "f6", 12: "50640102"})
    _assert_record(
        parse_log_event_packet(frame),
        BatteryInfo(
            raw=frame,
            log_type=0x15,
            time_stamp=TIME_STAMP,
            battery_level_millivolt=5354,
            temperature=-10,
            state_of_charge=80,
            state_of_health=100,
            battery_index=1,
            error=2,
        ),
    )


def test_product_message_type_and_data() -> None:
    """Built frame: a two-byte message type and five data bytes."""
    frame = _built(0x17, {5: TIME, 9: "3412", 11: "0102030405"})
    _assert_record(
        parse_log_event_packet(frame),
        ProductMessage(
            raw=frame,
            log_type=0x17,
            time_stamp=TIME_STAMP,
            msg_type=0x1234,
            data=(1, 2, 3, 4, 5),
        ),
    )


@pytest.mark.parametrize("log_type", [0x11, 0x14, 0x19], ids=["0x11", "0x14", "0x19"])
def test_a_log_type_with_no_decoder_gives_a_plain_record(log_type: int) -> None:
    """Built frames: AccessoryEvent has a name and no decoder, 0x14 and 0x19 no name."""
    frame = _built(log_type, {5: TIME, 12: "10"})
    record = parse_log_event_packet(frame)
    assert type(record) is LogRecord
    assert record.log_type == log_type
    assert record.raw == frame


@pytest.mark.parametrize(
    "frame",
    [
        KEYPAD_LOCK_BUTTON[:17],
        KEYPAD_LOCK_BUTTON + b"\x00",
        bytes.fromhex("bb02") + KEYPAD_LOCK_BUTTON[2:],
        bytes.fromhex("aa2d") + KEYPAD_LOCK_BUTTON[2:],
    ],
    ids=["17 bytes", "19 bytes", "starts BB 02", "starts AA 2D"],
)
def test_a_frame_that_is_no_get_log_reply_is_refused(frame: bytes) -> None:
    """The captured lock-button record, cut, extended, or with another header."""
    with pytest.raises(ValueError, match="Not an activity log reply"):
        parse_log_event_packet(frame)


@pytest.mark.parametrize(
    "case",
    [
        _captured_case(
            LockOperation,
            AUTO_RELOCK_LOCKED,
            op_code=OpCode.AUTO_RELOCK,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0x6ADDAD7B,
            battery_level=5920,
            temperature=25,
            error=63,
            battery_level_percent=63,
        ),
        _captured_case(
            LockOperation,
            INSIDE_BUTTON_LOCKED,
            op_code=OpCode.INSIDE_BUTTON,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0x6ADDEEA7,
            battery_level=5145,
            temperature=25,
            error=15,
            battery_level_percent=15,
        ),
        _captured_case(
            LockOperation,
            DOOR_STATE_CHANGED_SLOT_0,
            op_code=OpCode.DOOR_STATE_CHANGED,
            current_lock_state=LockStatus.UNLOCKED,
            key_slot=0,
            time_stamp=0x6ADDAE93,
            battery_level=5919,
            temperature=25,
            error=52,
            battery_level_percent=52,
        ),
        _captured_case(
            LockOperation,
            DOOR_STATE_CHANGED_SLOT_255,
            op_code=OpCode.DOOR_STATE_CHANGED,
            current_lock_state=LockStatus.LOCKED,
            key_slot=255,
            time_stamp=0x6AE03304,
            battery_level=5896,
            temperature=25,
            error=61,
            battery_level_percent=61,
        ),
        _case(
            KeypadWrongCode,
            0x02,
            {5: TIME, 9: "987654321fffff"},
            time_stamp=TIME_STAMP,
            keycode="987654321FFFFFFF",
        ),
        _captured_case(
            KeypadInfo,
            KEYPAD_INFO,
            time_stamp=0x6AE02F92,
            battery_level=195,
            stats=0x80000000,
        ),
        _case(CodeDeleted, 0x05, {5: TIME, 9: "07"}, time_stamp=TIME_STAMP, slot=7),
        _case(AllCodesDeleted, 0x06, {5: TIME}, time_stamp=TIME_STAMP),
        _captured_case(
            CodeOperation,
            CODE_OPERATION_SLOT_1,
            time_stamp=0x6AE02FFC,
            current_lock_state=LockStatus.UNLOCKED,
            current_door_state=DoorStatus.CLOSED,
            slot=1,
            error=0,
            battery_level=5899,
            battery_level_percent=71,
        ),
        _case(
            KeypadClose,
            0x08,
            {5: TIME, 9: "1234ffffffffff"},
            time_stamp=TIME_STAMP,
            keycode="1234FFFFFFFFFFFF",
        ),
        _case(FactoryReset, 0x09, {5: TIME, 9: "01"}, time_stamp=TIME_STAMP),
        _case(
            CodeDeletedv2, 0x0B, {5: TIME, 9: "eeff"}, time_stamp=TIME_STAMP, slot=65518
        ),
        _captured_case(
            CredentialOperation,
            CREDENTIAL_OPERATION_RF_SLOT_7,
            time_stamp=0x6ADDAD51,
            credential_type=LockCred.RF,
            slot=7,
            credential_error=None,
            current_lock_state=LockStatus.UNLOCKED,
            current_door_state=DoorStatus.CLOSED,
            battery_level=5919,
            battery_level_percent=66,
        ),
        _case(
            Tamper,
            0x0F,
            {5: TIME, 9: "0e00"},
            time_stamp=TIME_STAMP,
            tamper_type=TamperType.TOO_MANY_AUTH_ERRORS_FACE_SCAN_CLEARED,
        ),
        _case(
            Tamper, 0x0F, {5: TIME, 9: "0f01"}, time_stamp=TIME_STAMP, tamper_type=0x10F
        ),
        _case(
            BCMTest,
            0x10,
            {5: TIME, 9: "03", 10: "3412", 12: "04030201"},
            time_stamp=TIME_STAMP,
            error=3,
            seconds=0x1234,
            adv_counter=0x01020304,
        ),
        _case(
            HomekeyOperation,
            0x12,
            {5: TIME, 9: "15", 10: "02", 11: "3c", 12: "04030201"},
            time_stamp=TIME_STAMP,
            current_lock_state=LockStatus.LOCKED,
            battery_level_decivolt=60,
            key_id=0x01020304,
            error=2,
        ),
        _case(
            DPSInfo,
            0x13,
            {5: TIME, 9: "4b"},
            time_stamp=TIME_STAMP,
            battery_level_percent=75,
        ),
        _case(
            LocalEventNotification,
            0x16,
            {5: TIME, 9: "02"},
            time_stamp=TIME_STAMP,
            notification=2,
        ),
        _case(
            MatterMessage,
            0x18,
            {5: TIME, 9: "3412", 11: "0102030405"},
            time_stamp=TIME_STAMP,
            msg_type=0x1234,
            data=(1, 2, 3, 4, 5),
        ),
        _captured_case(
            DoorStateChanged,
            DOOR_CLOSED,
            time_stamp=0x6ADDAD5D,
            current_door_state=DoorStatus.CLOSED,
        ),
        _captured_case(
            DoorStateChanged,
            DOOR_OPENED,
            time_stamp=0x6ADDAD54,
            current_door_state=DoorStatus.OPENED,
        ),
        _case(
            DoorStateChanged,
            0x20,
            {5: TIME, 9: "02"},
            time_stamp=TIME_STAMP,
            current_door_state=DoorStatus.AJAR,
        ),
        _case(
            DoorStateChanged,
            0x20,
            {5: TIME, 9: "07"},
            time_stamp=TIME_STAMP,
            current_door_state=DoorStatus.UNKNOWN,
        ),
        _case(
            DialogReboot,
            0x40,
            {5: TIME, 9: "15", 10: "23", 11: "07", 12: "04030201"},
            time_stamp=TIME_STAMP,
            lock_state=LockStatus.LOCKED,
            door_state=DoorStatus.CLOSED,
            num_ble_conn=2,
            reboot_reason=3,
            task_id=7,
            pc=0x01020304,
        ),
    ],
    ids=_case_id,
)
def test_each_other_record_reads_its_layout(case: tuple[bytes, LogRecord]) -> None:
    """Captured frames where a lock sent one, built frames otherwise."""
    frame, expected = case
    _assert_record(parse_log_event_packet(frame), expected)


def _public_record_classes(cls: type[LogRecord]) -> set[type[LogRecord]]:
    found: set[type[LogRecord]] = set()
    for subclass in cls.__subclasses__():
        if not subclass.__name__.startswith("_"):
            found.add(subclass)
        found |= _public_record_classes(subclass)
    return found


def test_the_package_exports_the_decoder() -> None:
    """Every record class, enumeration, and name table is exported, with the parser."""
    records = _public_record_classes(LogRecord)
    assert len(records) == 26
    assert {cls.__name__ for cls in records} | {
        "LogRecord",
        "parse_log_event_packet",
        "LogType",
        "LockCred",
        "TamperType",
        "OpCode",
        "OperationError",
        "LOG_TYPE_NAMES_YALE",
        "OP_CODE_NAMES_YALE",
        "LOCK_STATE_NAMES_YALE",
        "CREDENTIAL_TYPE_NAMES_YALE",
        "DOOR_STATE_NAMES",
        "TAMPER_TYPE_NAMES_YALE",
    } <= set(yalexs_ble.__all__)


@pytest.mark.parametrize(
    ("table", "count", "expected"),
    [
        (
            LOG_TYPE_NAMES_YALE,
            27,
            {
                0x00: "LockOperation",
                0x01: "KeypadOpen",
                0x02: "KeypadWrongCode",
                0x03: "CodeAdded",
                0x04: "KeypadInfo",
                0x05: "CodeDeleted",
                0x06: "AllCodesDeleted",
                0x07: "CodeOperation",
                0x08: "KeypadClose",
                0x09: "FactoryReset",
                0x0A: "CodeAddedv2",
                0x0B: "CodeDeletedv2",
                0x0C: "CredentialAdded",
                0x0D: "CredentialDeleted",
                0x0E: "CredentialOperation",
                0x0F: "Tamper",
                0x10: "BCMTest",
                0x11: "AccessoryEvent",
                0x12: "HomekeyOperation",
                0x13: "DPSInfo",
                0x15: "BatteryInfo",
                0x16: "LocalEventNotification",
                0x17: "ProductMessage",
                0x18: "MatterMessage",
                0x20: "DoorStateChanged",
                0x40: "DialogReboot",
                0x80: "GetLogError",
            },
        ),
        (
            OP_CODE_NAMES_YALE,
            24,
            {
                0x00: "DoorStateChanged",
                0x01: "DoorManualStateChange",
                0x02: "KnockReceived",
                0x03: "BluetoothReceived",
                0x04: "Anomaly",
                0x05: "AutoRelock",
                0x06: "AutoRetry",
                0x07: "OverTemp",
                0x08: "BurnIn",
                0x09: "HomeKit",
                0x0A: "ZWave",
                0x0B: "OneTouch",
                0x0C: "InsideButton",
                0x0E: "DoorSense",
                0x0F: "LowBatAutoUnlock",
                0x10: "EscapeUnlock",
                0x11: "UWB",
                0x12: "ExternalModuleOperation",
                0x13: "ShabbatMode",
                0x14: "ScheduledOperation",
                0x15: "Matter",
                0x17: "DoublePress",
                0x18: "Aliro",
                0xFF: "Undefined",
            },
        ),
        (
            LOCK_STATE_NAMES_YALE,
            17,
            {
                0x00: "Init",
                0x01: "Calibrating",
                0x02: "Unlocking",
                0x03: "Unlocked",
                0x04: "Locking",
                0x05: "Locked",
                0x06: "PolDiscovery",
                0x07: "UnknownStaticPosition",
                0x09: "Unlatching",
                0x0A: "Unlatched",
                0x0B: "Integrated_Other_System",
                0x0C: "SecureMode",
                0x0D: "Master code not set",
                0x0E: "Default master code set",
                0x0F: "Lock not handed",
                0x10: "shabbatmode",
                0x11: "InternalForcedLocked",
            },
        ),
        (
            CREDENTIAL_TYPE_NAMES_YALE,
            8,
            {
                0x00: "pin",
                0x01: "rf",
                0x02: "finger",
                0x04: "face",
                0x05: "inside_door_pin",
                0x06: "palm",
                0x08: "aliro",
                0xFF: "All",
            },
        ),
        (
            DOOR_STATE_NAMES,
            5,
            {0x00: "Init", 0x01: "Closed", 0x02: "Ajar", 0x03: "Open", 0x04: "Unknown"},
        ),
        (
            TAMPER_TYPE_NAMES_YALE,
            15,
            {
                0x00: "TamperClear",
                0x01: "TamperTooManyAuthErrors",
                0x02: "TamperFrontRemoved",
                0x03: "TamperDoorForcedOpen",
                0x04: "TamperHighTemperature",
                0x05: "TamperTooManyAuthErrorsPIN",
                0x06: "TamperTooManyAuthErrorsPINCleared",
                0x07: "TamperTooManyAuthErrorsRFID",
                0x08: "TamperTooManyAuthErrorsRFIDCleared",
                0x09: "TamperTooManyAuthErrorsFingerprint",
                0x0A: "TamperTooManyAuthErrorsFingerprintCleared",
                0x0B: "TamperTooManyAuthErrorsPalm",
                0x0C: "TamperTooManyAuthErrorsPalmCleared",
                0x0D: "TamperTooManyAuthErrorsFaceScan",
                0x0E: "TamperTooManyAuthErrorsFaceScanCleared",
            },
        ),
    ],
    ids=[
        "log type",
        "opCode",
        "lock state",
        "credential type",
        "door state",
        "tamper type",
    ],
)
def test_name_tables_hold_the_vendor_strings(
    table: dict[int, str], count: int, expected: dict[int, str]
) -> None:
    """Each table holds exactly the vendor's entries."""
    assert len(table) == count
    assert table == expected


@pytest.mark.parametrize(
    ("enumeration", "names"),
    [
        (LogType, LOG_TYPE_NAMES_YALE),
        (OpCode, OP_CODE_NAMES_YALE),
        (LockCred, CREDENTIAL_TYPE_NAMES_YALE),
        (TamperType, TAMPER_TYPE_NAMES_YALE),
    ],
    ids=["LogType", "OpCode", "LockCred", "TamperType"],
)
def test_enumeration_members_are_the_named_values(
    enumeration: type[IntEnum], names: dict[int, str]
) -> None:
    """A member has the value and the vendor's term less a prefix of the type's name."""
    assert {member.value for member in enumeration} == set(names)
    for name, member in enumeration.__members__.items():
        vendor_name = names[member].replace("_", "").lower()
        member_name = name.replace("_", "").lower()
        assert vendor_name.endswith(member_name)
        prefix = vendor_name.removesuffix(member_name)
        assert enumeration.__name__.lower().startswith(prefix)


def test_log_type_get_log_error_is_the_value_that_ends_the_log() -> None:
    """LogType.GET_LOG_ERROR takes its value from LOG_TYPE_GET_LOG_ERROR."""
    assert LogType.GET_LOG_ERROR == LOG_TYPE_GET_LOG_ERROR
