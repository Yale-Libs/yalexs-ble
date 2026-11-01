from collections.abc import Callable
from enum import Enum

import pytest

import yalexs_ble
from yalexs_ble.activity_log import (
    AllCodesDeleted,
    BatteryInfo,
    BCMTest,
    BulkOperation,
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
    InvalidCredentialEntry,
    KeypadClose,
    KeypadInfo,
    KeypadOpen,
    KeypadWrongCode,
    LocalEventNotification,
    LockDeadboltJam,
    LockOperatingMode,
    LockOperation,
    LockOperationBlame,
    LogRecord,
    MatterEvent,
    MatterMessage,
    OutOfScheduleCredential,
    ProductMessage,
    Tamper,
    parse_log_event_packet,
)
from yalexs_ble.const import (
    RESPONSE_FRAME_LEN,
    AugustLockOperationSource,
    AugustLogType,
    BulkOperationType,
    CredType,
    DoorStatus,
    JamState,
    LockCred,
    LockOperationCredType,
    LockOperationStatus,
    LockOperationType,
    LockStatus,
    LogBrand,
    LogType,
    MatterEventType,
    MatterRegistrationStatus,
    OpCode,
    OperatingMode,
    OutOfScheduleType,
    TamperType,
)
from yalexs_ble.util import _simple_checksum

# Captured GetLog replies.
KEYPAD_UNLOCK_SLOT_205 = bytes.fromhex("bb2d0072079aa176a333004bea14cd000200")
KEYPAD_UNLOCK_SLOT_200 = bytes.fromhex("bb2d00a4077a9e76a333004cdf14c8000200")
KEYPAD_UNLOCK_SLOT_UNKNOWN = bytes.fromhex("bb2d00d007299c76a3330049e414eeff0200")
KEYPAD_LOCK_BUTTON = bytes.fromhex("bb2d0056000b0500a0a176a3df14194a0200")
END_OF_LOG = bytes.fromhex("bb2d00de800200009aa176a3a8a176a30200")

# The captured frames both brand comparisons decode.
CAPTURED = [
    KEYPAD_UNLOCK_SLOT_205,
    KEYPAD_UNLOCK_SLOT_200,
    KEYPAD_UNLOCK_SLOT_UNKNOWN,
    KEYPAD_LOCK_BUTTON,
    END_OF_LOG,
]

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

# Bytes 5 to 15 of a built frame whose two decodes a test compares.
BODY = "0102030405060708090a0b"

# The fields of a built credential record: rf, slot 65518, unlocked and closed.
CREDENTIAL_FIELDS = {5: TIME, 9: "01", 10: "eeff", 12: "13", 13: "4b", 14: "ea14"}


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
    record: Callable[..., LogRecord], frame: bytes, **values: object
) -> tuple[bytes, LogRecord]:
    """A captured or built frame and the record its bytes decode to."""
    return frame, record(raw=frame, **values)


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
            op_code=OpCode.ONE_TOUCH,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0xA376A1A0,
            battery_level=5343,
            temperature=25,
            battery_level_percent=74,
        ),
    )


def test_captured_get_log_error_record() -> None:
    """Captured frame: the record an empty log answers with."""
    _assert_record(
        parse_log_event_packet(END_OF_LOG),
        GetLogError(
            raw=END_OF_LOG,
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
    ("log_type", "brand", "record_class"),
    [
        (0x0C, LogBrand.YALE, CredentialAdded),
        (0x0D, LogBrand.YALE, CredentialDeleted),
        (0x0E, LogBrand.YALE, CredentialOperation),
        (0x0C, LogBrand.AUGUST, CredentialAdded),
        (0x0D, LogBrand.AUGUST, CredentialDeleted),
    ],
    ids=[
        "CredentialAdded, Yale",
        "CredentialDeleted, Yale",
        "CredentialOperation, Yale",
        "CredentialAdded, August",
        "CredentialDeleted, August",
    ],
)
def test_credential_records_with_states(
    log_type: int, brand: LogBrand, record_class: Callable[..., LogRecord]
) -> None:
    """Built frames: byte 12 holds the two states and the slot is two bytes."""
    frame = _built(log_type, CREDENTIAL_FIELDS)
    _assert_record(
        parse_log_event_packet(frame, brand),
        record_class(
            raw=frame,
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


def test_credential_type_without_a_member_is_kept_as_unknown() -> None:
    """Built frame: credential type 3 has no name."""
    record = parse_log_event_packet(_built(0x0E, {9: "03"}))
    assert isinstance(record, CredentialOperation)
    assert record.credential_type is LockCred(3)
    assert record.credential_type == 3
    assert record.credential_type.name == "UNKNOWN_03"


def test_an_open_enumeration_refuses_a_value_that_is_not_an_int() -> None:
    """Only an int becomes an UNKNOWN member; anything else is a ValueError."""
    with pytest.raises(ValueError, match="not a valid LockCred"):
        LockCred("03")  # type: ignore[arg-type]


def test_lock_operation_with_an_unnamed_op_code_is_returned() -> None:
    """Built frame: opCode 0x0D has no named member and is kept as UNKNOWN_0D."""
    record = parse_log_event_packet(_built(0x00, {5: "0d", 6: "05"}))
    assert isinstance(record, LockOperation)
    assert record.op_code is OpCode(0x0D)
    assert record.op_code.value == 13
    assert record.op_code.name == "UNKNOWN_0D"


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
        KeypadOpen(raw=frame, time_stamp=TIME_STAMP, keycode="123456FFFFFFFFFF"),
    )


def test_code_added_slot_and_code() -> None:
    """Built frame: the six code bytes are padded to 16 hex digits."""
    frame = _built(0x03, {5: TIME, 9: "07", 10: "1234ffffffff"})
    _assert_record(
        parse_log_event_packet(frame),
        CodeAdded(
            raw=frame,
            time_stamp=TIME_STAMP,
            slot=7,
            code="1234FFFFFFFFFFFF",
        ),
    )


@pytest.mark.parametrize(
    ("brand", "code"),
    [(LogBrand.YALE, "000000000001E240"), (LogBrand.AUGUST, "123456")],
    ids=["Yale", "August"],
)
def test_code_added_v2_code_as_the_brand_prints_it(brand: LogBrand, code: str) -> None:
    """Built frame: the five code bytes are one little-endian number, hex or decimal."""
    frame = _built(0x0A, {5: TIME, 9: "0102", 11: "40e2010000"})
    _assert_record(
        parse_log_event_packet(frame, brand),
        CodeAddedv2(raw=frame, time_stamp=TIME_STAMP, slot=0x0201, code=code),
    )


def test_battery_info_temperature_is_signed() -> None:
    """Built frame: byte 11 is a signed number."""
    frame = _built(0x15, {5: TIME, 9: "ea14", 11: "f6", 12: "50640102"})
    _assert_record(
        parse_log_event_packet(frame),
        BatteryInfo(
            raw=frame,
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
            time_stamp=TIME_STAMP,
            msg_type=0x1234,
            data=(1, 2, 3, 4, 5),
        ),
    )


@pytest.mark.parametrize(
    ("brand", "log_type"),
    [
        (LogBrand.YALE, 0x11),
        (LogBrand.YALE, 0x14),
        (LogBrand.YALE, 0x19),
        (LogBrand.AUGUST, 0x11),
        (LogBrand.AUGUST, 0x1A),
    ],
    ids=["Yale 0x11", "Yale 0x14", "Yale 0x19", "August 0x11", "August 0x1A"],
)
def test_a_log_type_with_no_decoder_gives_a_plain_record(
    brand: LogBrand, log_type: int
) -> None:
    """Built frames: AccessoryEvent has a name and no decoder, the others no name."""
    frame = _built(log_type, {5: TIME, 12: "10"})
    record = parse_log_event_packet(frame, brand)
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


def test_a_bytearray_frame_decodes_to_a_record_holding_bytes() -> None:
    """Captured frame in a bytearray: the record keeps a bytes copy of it."""
    frame = bytearray(KEYPAD_LOCK_BUTTON)
    # A bytearray is outside the bytes signature, and the copy keeps a later
    # change to it out of the record.
    record = parse_log_event_packet(frame)  # type: ignore[arg-type]
    assert type(record.raw) is bytes
    assert record.raw == KEYPAD_LOCK_BUTTON
    frame[0x08] ^= 0xFF
    _assert_record(record, parse_log_event_packet(KEYPAD_LOCK_BUTTON))


@pytest.mark.parametrize(
    "case",
    [
        _case(
            LockOperation,
            AUTO_RELOCK_LOCKED,
            op_code=OpCode.AUTO_RELOCK,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0x6ADDAD7B,
            battery_level=5920,
            temperature=25,
            battery_level_percent=63,
        ),
        _case(
            LockOperation,
            INSIDE_BUTTON_LOCKED,
            op_code=OpCode.INSIDE_BUTTON,
            current_lock_state=LockStatus.LOCKED,
            key_slot=0,
            time_stamp=0x6ADDEEA7,
            battery_level=5145,
            temperature=25,
            battery_level_percent=15,
        ),
        _case(
            LockOperation,
            DOOR_STATE_CHANGED_SLOT_0,
            op_code=OpCode.DOOR_STATE_CHANGED,
            current_lock_state=LockStatus.UNLOCKED,
            key_slot=0,
            time_stamp=0x6ADDAE93,
            battery_level=5919,
            temperature=25,
            battery_level_percent=52,
        ),
        _case(
            LockOperation,
            DOOR_STATE_CHANGED_SLOT_255,
            op_code=OpCode.DOOR_STATE_CHANGED,
            current_lock_state=LockStatus.LOCKED,
            key_slot=255,
            time_stamp=0x6AE03304,
            battery_level=5896,
            temperature=25,
            battery_level_percent=61,
        ),
        _case(
            KeypadWrongCode,
            _built(0x02, {5: TIME, 9: "987654321fffff"}),
            time_stamp=TIME_STAMP,
            keycode="987654321FFFFFFF",
        ),
        _case(
            KeypadInfo,
            KEYPAD_INFO,
            time_stamp=0x6AE02F92,
            battery_level=195,
            stats=0x80000000,
        ),
        _case(
            CodeDeleted, _built(0x05, {5: TIME, 9: "07"}), time_stamp=TIME_STAMP, slot=7
        ),
        _case(AllCodesDeleted, _built(0x06, {5: TIME}), time_stamp=TIME_STAMP),
        _case(
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
            _built(0x08, {5: TIME, 9: "1234ffffffffff"}),
            time_stamp=TIME_STAMP,
            keycode="1234FFFFFFFFFFFF",
        ),
        _case(FactoryReset, _built(0x09, {5: TIME, 9: "01"}), time_stamp=TIME_STAMP),
        _case(
            CodeDeletedv2,
            _built(0x0B, {5: TIME, 9: "eeff"}),
            time_stamp=TIME_STAMP,
            slot=65518,
        ),
        _case(
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
            _built(0x0F, {5: TIME, 9: "0e00"}),
            time_stamp=TIME_STAMP,
            tamper_type=TamperType.TOO_MANY_AUTH_ERRORS_FACE_SCAN_CLEARED,
        ),
        _case(
            Tamper,
            _built(0x0F, {5: TIME, 9: "0f01"}),
            time_stamp=TIME_STAMP,
            tamper_type=TamperType(0x10F),
        ),
        _case(
            BCMTest,
            _built(0x10, {5: TIME, 9: "03", 10: "3412", 12: "04030201"}),
            time_stamp=TIME_STAMP,
            error=3,
            seconds=0x1234,
            adv_counter=0x01020304,
        ),
        _case(
            HomekeyOperation,
            _built(0x12, {5: TIME, 9: "15", 10: "02", 11: "3c", 12: "04030201"}),
            time_stamp=TIME_STAMP,
            current_lock_state=LockStatus.LOCKED,
            battery_level_decivolt=60,
            key_id=0x01020304,
            error=2,
        ),
        _case(
            DPSInfo,
            _built(0x13, {5: TIME, 9: "4b"}),
            time_stamp=TIME_STAMP,
            battery_level_percent=75,
        ),
        _case(
            LocalEventNotification,
            _built(0x16, {5: TIME, 9: "02"}),
            time_stamp=TIME_STAMP,
            notification=2,
        ),
        _case(
            MatterMessage,
            _built(0x18, {5: TIME, 9: "3412", 11: "0102030405"}),
            time_stamp=TIME_STAMP,
            msg_type=0x1234,
            data=(1, 2, 3, 4, 5),
        ),
        _case(
            DoorStateChanged,
            DOOR_CLOSED,
            time_stamp=0x6ADDAD5D,
            current_door_state=DoorStatus.CLOSED,
        ),
        _case(
            DoorStateChanged,
            DOOR_OPENED,
            time_stamp=0x6ADDAD54,
            current_door_state=DoorStatus.OPENED,
        ),
        _case(
            DoorStateChanged,
            _built(0x20, {5: TIME, 9: "02"}),
            time_stamp=TIME_STAMP,
            current_door_state=DoorStatus.AJAR,
        ),
        _case(
            DoorStateChanged,
            _built(0x20, {5: TIME, 9: "07"}),
            time_stamp=TIME_STAMP,
            current_door_state=DoorStatus.UNKNOWN,
        ),
        _case(
            DialogReboot,
            _built(0x40, {5: TIME, 9: "15", 10: "23", 11: "07", 12: "04030201"}),
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


def test_with_no_brand_a_record_reads_in_yales_numbering() -> None:
    """Captured and built frames decode alike with no brand and with LogBrand.YALE."""
    frames = [*CAPTURED, *(_built(log_type, {5: BODY}) for log_type in range(0x100))]
    for frame in frames:
        _assert_record(
            parse_log_event_packet(frame),
            parse_log_event_packet(frame, LogBrand.YALE),
        )


def test_a_log_type_with_the_same_decoder_in_both_brands_reads_alike() -> None:
    """Captured and built frames for each log_type the two brands decode alike."""
    frames = [*CAPTURED, _built(0x0E, CREDENTIAL_FIELDS)]
    alike = [*range(0x0A), *range(0x0B, 0x14), 0x20, 0x40, 0x80]
    frames += [_built(log_type, {5: BODY}) for log_type in alike]
    for frame in frames:
        _assert_record(
            parse_log_event_packet(frame, LogBrand.AUGUST),
            parse_log_event_packet(frame, LogBrand.YALE),
        )


@pytest.mark.parametrize(
    ("log_type", "brand", "record_class"),
    [
        (0x17, LogBrand.YALE, ProductMessage),
        (0x17, LogBrand.AUGUST, InvalidCredentialEntry),
        (0x15, LogBrand.YALE, BatteryInfo),
        (0x15, LogBrand.AUGUST, LogRecord),
    ],
    ids=["0x17, Yale", "0x17, August", "0x15, Yale", "0x15, August"],
)
def test_a_record_the_brands_number_differently_reads_in_the_brands_numbering(
    log_type: int, brand: LogBrand, record_class: type[LogRecord]
) -> None:
    """Built frames: 0x17 is one record under Yale and another under August."""
    frame = _built(log_type, {5: BODY})
    record = parse_log_event_packet(frame, brand)
    assert type(record) is record_class
    assert record.raw == frame


@pytest.mark.parametrize(
    "case",
    [
        _case(
            LockOperatingMode,
            _built(0x14, {5: TIME, 9: "02"}),
            time_stamp=TIME_STAMP,
            lock_op_mode=OperatingMode.PRIVACY,
        ),
        _case(
            LockDeadboltJam,
            _built(0x16, {5: TIME, 9: "ea14", 11: "01"}),
            time_stamp=TIME_STAMP,
            battery_level=5354,
            jam_state=JamState.OCCURS_DURING_UNLOCKING,
        ),
        _case(
            InvalidCredentialEntry,
            _built(0x17, {5: TIME, 9: "0110", 11: "df14"}),
            time_stamp=TIME_STAMP,
            cred_type=CredType.ENTRY_APPLE_HOMEKEYS_NFC,
            battery_level=5343,
        ),
        _case(
            InvalidCredentialEntry,
            _built(0x17, {5: TIME, 9: "0300", 11: "df14"}),
            time_stamp=TIME_STAMP,
            cred_type=CredType(0x0003),
            battery_level=5343,
        ),
        _case(
            MatterEvent,
            _built(0x18, {5: TIME, 9: "01", 10: "03", 11: "4913"}),
            time_stamp=TIME_STAMP,
            event_type=MatterEventType.FABRIC_ADDED,
            fabric_index=3,
            fabric_vendor_id=0x1349,
            registration_status=None,
        ),
        _case(
            LockOperationBlame,
            _built(
                0x19,
                {5: TIME, 9: "01", 10: "02", 11: "02", 12: "0700", 14: "09", 15: "03"},
            ),
            time_stamp=TIME_STAMP,
            op_type=LockOperationType.UNLOCK,
            op_source=AugustLockOperationSource.KEYPAD,
            op_lock_cred_types=LockOperationCredType.PIN,
            key_slot=None,
            node_id=None,
            user_slot=7,
            fabric_index=None,
            op_status=LockOperationStatus.JAMMED,
        ),
        _case(
            OutOfScheduleCredential,
            _built(0x1D, {5: TIME, 9: "0000", 11: "df14", 13: "01"}),
            time_stamp=TIME_STAMP,
            cred_type=CredType.ENTRY_KEYPAD_PINCODE,
            battery_level=5343,
            violation_type=OutOfScheduleType.EXPIRED,
        ),
        _case(
            BulkOperation,
            _built(0x1E, {5: TIME, 9: "01", 10: "02", 11: "0500"}),
            time_stamp=TIME_STAMP,
            bulk_op_type=BulkOperationType.USERS_CLEAR,
            fabric_index=2,
            lock_cred_type=None,
            number_of_removed_users=5,
        ),
    ],
    ids=_case_id,
)
def test_each_august_record_reads_its_layout(case: tuple[bytes, LogRecord]) -> None:
    """Built frames for the seven records only August's numbering decodes."""
    frame, expected = case
    _assert_record(parse_log_event_packet(frame, LogBrand.AUGUST), expected)


def test_a_value_only_yales_app_names_keeps_its_member_under_august() -> None:
    """Built frame: opCode 0x13 and lock state 0x11 have no name in August's app."""
    record = parse_log_event_packet(_built(0x00, {5: "13", 6: "11"}), LogBrand.AUGUST)
    assert isinstance(record, LockOperation)
    assert record.op_code is OpCode.SHABBAT_MODE
    assert record.current_lock_state is LockStatus.INTERNAL_FORCED_LOCKED


@pytest.mark.parametrize(
    ("value", "op_source"),
    [
        (0x01, AugustLockOperationSource.BLE),
        (0x02, AugustLockOperationSource.KEYPAD),
        (0x03, AugustLockOperationSource.LOCK_BUTTON),
        (0x04, AugustLockOperationSource.AUTORELOCK),
        (0x05, AugustLockOperationSource.MANUAL),
        (0x06, AugustLockOperationSource.MATTER),
        (0x07, AugustLockOperationSource.ZWAVE),
        (0x08, AugustLockOperationSource(0x08)),
    ],
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
)
def test_lock_operation_blame_op_source_is_its_member(
    value: int, op_source: AugustLockOperationSource
) -> None:
    """Built frames: byte 10 is the member, or UNKNOWN_xx when it has no name."""
    frame = _built(0x19, {10: f"{value:02x}"})
    record = parse_log_event_packet(frame, LogBrand.AUGUST)
    assert isinstance(record, LockOperationBlame)
    assert record.op_source == op_source
    assert type(record.op_source) is type(op_source)


@pytest.mark.parametrize(
    ("op_source", "key_slot", "node_id", "user_slot", "fabric_index"),
    [
        (AugustLockOperationSource.MANUAL, None, None, None, None),
        (AugustLockOperationSource.ZWAVE, 0x1234, 7, None, None),
        (AugustLockOperationSource.KEYPAD, None, None, 0x1234, None),
        (AugustLockOperationSource.MATTER, None, None, 0x1234, 7),
        (0x08, None, None, 0x1234, None),
    ],
    ids=["MANUAL", "ZWAVE", "KEYPAD", "MATTER", "8"],
)
def test_lock_operation_blame_slot_fields_follow_op_source(
    op_source: AugustLockOperationSource | int,
    key_slot: int | None,
    node_id: int | None,
    user_slot: int | None,
    fabric_index: int | None,
) -> None:
    """Built frames: op_source decides how bytes 12 to 14 are read."""
    frame = _built(0x19, {10: f"{op_source:02x}", 12: "341207"})
    record = parse_log_event_packet(frame, LogBrand.AUGUST)
    assert isinstance(record, LockOperationBlame)
    assert (
        record.key_slot,
        record.node_id,
        record.user_slot,
        record.fabric_index,
    ) == (key_slot, node_id, user_slot, fabric_index)


@pytest.mark.parametrize(
    ("cred_types", "expected"),
    [
        ("00", None),
        ("01", None),
        ("06", LockOperationCredType.PIN | LockOperationCredType.RF_ID),
    ],
)
def test_lock_operation_blame_credential_types_are_a_bit_mask(
    cred_types: str, expected: LockOperationCredType | None
) -> None:
    """Built frames: bits 1 to 7 of byte 11, and None when none of them is set."""
    frame = _built(0x19, {11: cred_types})
    record = parse_log_event_packet(frame, LogBrand.AUGUST)
    assert isinstance(record, LockOperationBlame)
    assert record.op_lock_cred_types == expected
    assert type(record.op_lock_cred_types) is type(expected)


@pytest.mark.parametrize(
    ("event_type", "fabric_index", "fabric_vendor_id", "registration_status"),
    [
        (0, None, None, MatterRegistrationStatus.FAIL),
        (1, 1, 0x1234, None),
        (2, 1, 0x1234, None),
        (3, None, None, None),
    ],
)
def test_matter_event_fields_follow_event_type(
    event_type: int,
    fabric_index: int | None,
    fabric_vendor_id: int | None,
    registration_status: MatterRegistrationStatus | None,
) -> None:
    """Built frames: event_type decides how bytes 10 to 12 are read."""
    frame = _built(0x18, {9: f"{event_type:02x}", 10: "013412"})
    record = parse_log_event_packet(frame, LogBrand.AUGUST)
    assert isinstance(record, MatterEvent)
    assert record.event_type == event_type
    assert record.fabric_index == fabric_index
    assert record.fabric_vendor_id == fabric_vendor_id
    assert record.registration_status is registration_status


@pytest.mark.parametrize(
    ("body", "fabric_index", "lock_cred_type", "number_of_removed_users"),
    [
        ("0103010200", 3, None, 0x0201),
        ("01ff050000", None, None, 5),
        ("0204030102", 3, LockOperationCredType.RF_ID, 0x0201),
        ("0204ff0500", None, LockOperationCredType.RF_ID, 5),
        ("0201030500", 3, LockOperationCredType(1), 5),
        ("0302020500", None, None, None),
    ],
    ids=[
        "users clear",
        "users clear, no fabric",
        "creds clear",
        "creds clear, no fabric",
        "creds clear, unnamed credential type",
        "type 3",
    ],
)
def test_bulk_operation_fields_follow_bulk_op_type(
    body: str,
    fabric_index: int | None,
    lock_cred_type: LockOperationCredType | None,
    number_of_removed_users: int | None,
) -> None:
    """Built frames: body is bytes 9 to 13, starting with bulk_op_type."""
    record = parse_log_event_packet(_built(0x1E, {9: body}), LogBrand.AUGUST)
    assert isinstance(record, BulkOperation)
    assert record.bulk_op_type == int(body[:2], 16)
    assert record.fabric_index == fabric_index
    assert record.lock_cred_type == lock_cred_type
    assert type(record.lock_cred_type) is type(lock_cred_type)
    assert record.number_of_removed_users == number_of_removed_users


def _public_record_classes(cls: type[LogRecord]) -> set[type[LogRecord]]:
    found: set[type[LogRecord]] = set()
    for subclass in cls.__subclasses__():
        if not subclass.__name__.startswith("_"):
            found.add(subclass)
        found |= _public_record_classes(subclass)
    return found


def test_the_package_exports_the_decoder() -> None:
    """Every record class and enumeration is exported with the parser; the name
    tables stay in yalexs_ble.const."""
    records = _public_record_classes(LogRecord)
    assert {cls.__name__ for cls in records} | {
        "LogRecord",
        "parse_log_event_packet",
        "LogBrand",
        "LogType",
        "AugustLogType",
        "LockCred",
        "TamperType",
        "OpCode",
        "OperationError",
        "OperatingMode",
        "JamState",
        "CredType",
        "OutOfScheduleType",
        "MatterEventType",
        "MatterRegistrationStatus",
        "LockOperationType",
        "AugustLockOperationSource",
        "LockOperationCredType",
        "LockOperationStatus",
        "BulkOperationType",
    } <= set(yalexs_ble.__all__)


@pytest.mark.parametrize(
    ("enumeration", "expected"),
    [
        (
            AugustLogType,
            {
                "LOCK_OPERATION": 0x00,
                "KEYPAD_OPEN": 0x01,
                "KEYPAD_WRONG_CODE": 0x02,
                "CODE_ADDED": 0x03,
                "KEYPAD_INFO": 0x04,
                "CODE_DELETED": 0x05,
                "ALL_CODES_DELETED": 0x06,
                "CODE_OPERATION": 0x07,
                "KEYPAD_CLOSE": 0x08,
                "FACTORY_RESET": 0x09,
                "CODE_ADDED_V2": 0x0A,
                "CODE_DELETED_V2": 0x0B,
                "EXTERNAL_CREDENTIAL_ADDED": 0x0C,
                "EXTERNAL_CREDENTIAL_DELETED": 0x0D,
                "CREDENTIAL_OPERATION": 0x0E,
                "TAMPER": 0x0F,
                "BCM_TEST": 0x10,
                "ACCESSORY_EVENT": 0x11,
                "HOMEKEY_OPERATION": 0x12,
                "DPS_INFO": 0x13,
                "LOCK_OPERATING_MODE": 0x14,
                "WIFI_INFO": 0x15,
                "LOCK_DEADBOLT_JAM": 0x16,
                "INVALID_CREDENTIAL_ENTRY": 0x17,
                "MATTER_EVENT": 0x18,
                "LOCK_OPERATION_BLAME": 0x19,
                "OUT_OF_SCHEDULE_CREDENTIAL": 0x1D,
                "BULK_OPERATION": 0x1E,
                "DOOR_STATE_CHANGED": 0x20,
                "DIALOG_REBOOT": 0x40,
                "GET_LOG_ERROR": 0x80,
            },
        ),
        (
            OperatingMode,
            {
                "NORMAL": 0,
                "VACATION": 1,
                "PRIVACY": 2,
                "PASSAGE": 3,
            },
        ),
        (
            JamState,
            {"OCCURS_DURING_LOCKING": 0, "OCCURS_DURING_UNLOCKING": 1},
        ),
        (
            CredType,
            {
                "ENTRY_KEYPAD_PINCODE": 0x0000,
                "ENTRY_KEYPAD_FINGERPRINT": 0x0002,
                "ENTRY_AUX_KEYPAD_PINCODE": 0x0100,
                "ENTRY_AUX_KEYPAD_FINGERPRINT": 0x0102,
                "ENTRY_APPLE_HOMEKEYS_NFC": 0x1001,
                "ENTRY_SAMSUNG_WALLET_NFC": 0x1101,
                "ENTRY_GOOGLE_WALLET_NFC": 0x1201,
                "ENTRY_RFID_CARD_MIFARE_DESFIRE_EV1_NFC": 0x2101,
                "ENTRY_RFID_CARD_MIFARE_DESFIRE_EV2_NFC": 0x2201,
                "ENTRY_RFID_CARD_MIFARE_DESFIRE_EV3_NFC": 0x2301,
                "ENTRY_MATTER_FABRIC_MATTER": 0x3003,
                "ENTRY_UNKNOWN": 0xFFFF,
            },
        ),
        (
            OutOfScheduleType,
            {"PREMATURE": 0, "EXPIRED": 1},
        ),
        (
            MatterEventType,
            {
                "REGISTRATION": 0,
                "FABRIC_ADDED": 1,
                "FABRIC_REMOVED": 2,
            },
        ),
        (MatterRegistrationStatus, {"OK": 0, "FAIL": 1, "UNKNOWN": 0xFF}),
        (LockOperationType, {"LOCK": 0, "UNLOCK": 1}),
        (
            AugustLockOperationSource,
            {
                "BLE": 1,
                "KEYPAD": 2,
                "LOCK_BUTTON": 3,
                "AUTORELOCK": 4,
                "MANUAL": 5,
                "MATTER": 6,
                "ZWAVE": 7,
            },
        ),
        (
            LockOperationCredType,
            {
                "PIN": 0x02,
                "RF_ID": 0x04,
                "FINGERPRINT": 0x08,
                "RF_ID_HID": 0x10,
                "FACE": 0x20,
                "INSIDE_DOOR_PIN": 0x40,
                "BLE": 0x80,
            },
        ),
        (
            LockOperationStatus,
            {
                "SUCCESS": 0,
                "FAIL": 1,
                "LOW_BATTERY": 2,
                "JAMMED": 3,
                "NON_CALIBRATED": 4,
                "TIMEOUT": 5,
                "INVALID_CRED": 6,
                "DISABLED_USER": 7,
                "OUTSIDE_SCHEDULE": 8,
                "NON_ACCESS": 9,
            },
        ),
        (BulkOperationType, {"USERS_CLEAR": 1, "CREDS_CLEAR": 2}),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_august_enumerations_hold_the_vendor_values(
    enumeration: type[Enum], expected: dict[str, int]
) -> None:
    """Each enumeration has exactly these members."""
    assert {member.name: member.value for member in enumeration} == expected


def test_august_log_type_matches_log_type_where_the_brands_agree() -> None:
    """The two numberings name 21 log_type values alike and six differently."""
    yale = {member.value: member.name for member in LogType}
    august = {member.value: member.name for member in AugustLogType}
    alike = {
        value for value in yale.keys() & august.keys() if yale[value] == august[value]
    }
    assert alike == {*range(0x0C), *range(0x0E, 0x14), 0x20, 0x40, 0x80}
    assert (yale.keys() & august.keys()) - alike == {0x0C, 0x0D, 0x15, 0x16, 0x17, 0x18}
