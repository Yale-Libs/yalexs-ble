"""Decode the records of the lock's activity log."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self

from .const import (
    GET_LOG_ERROR_DIRTY_BIT,
    RESPONSE_FRAME_LEN,
    VALUE_TO_DOOR_STATUS,
    VALUE_TO_LOCK_STATUS,
    AugustLockOperationSource,
    AugustLogType,
    BulkOperationType,
    Commands,
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
from .util import _bytes_to_int, _named

# August's app emits no fabricIndex for a BulkOperation whose fabric byte
# is 0xFF.
_NO_FABRIC_INDEX = 0xFF

# LockOperationCredType is an IntFlag, which answers an unnamed value with a
# pseudo-member and no ValueError, so _named cannot serve it.
_LOCK_OPERATION_CRED_TYPES = {member.value: member for member in LockOperationCredType}


def _u16(raw: bytes, offset: int) -> int:
    return _bytes_to_int(raw[offset : offset + 2])


def _u32(raw: bytes, offset: int) -> int:
    return _bytes_to_int(raw[offset : offset + 4])


# raw keeps a state byte that has no member.
def _lock_state(value: int) -> LockStatus:
    return VALUE_TO_LOCK_STATUS.get(value, LockStatus.UNKNOWN)


def _door_state(value: int) -> DoorStatus:
    return VALUE_TO_DOOR_STATUS.get(value, DoorStatus.UNKNOWN)


def _code(code: bytes) -> str:
    """Format a keypad code as 16 hex digits, first byte first, padded with F."""
    return code.hex().upper().ljust(16, "F")


@dataclass(frozen=True)
class LogRecord:
    """One record of the activity log, and the value of a record with no decoder."""

    raw: bytes
    log_type: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        """Build the record from the 18 bytes of a reply."""
        return cls(raw, raw[4])


@dataclass(frozen=True)
class LockOperation(LogRecord):
    """A LockOperation record (log_type 0x00)."""

    op_code: OpCode | int
    current_lock_state: LockStatus
    key_slot: int
    time_stamp: int
    battery_level: int
    temperature: int
    error: int
    battery_level_percent: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            op_code=_named(OpCode, raw[5]),
            current_lock_state=_lock_state(raw[6]),
            key_slot=raw[7],
            time_stamp=_u32(raw, 8),
            battery_level=_u16(raw, 12),
            temperature=raw[14],
            # The vendor's parser emits byte 15 under both keys.
            error=raw[15],
            battery_level_percent=raw[15],
        )


@dataclass(frozen=True)
class _KeypadCode(LogRecord):
    """The layout of the KeypadOpen, KeypadWrongCode, and KeypadClose records."""

    time_stamp: int
    keycode: str

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5), keycode=_code(raw[9:16]))


@dataclass(frozen=True)
class KeypadOpen(_KeypadCode):
    """A KeypadOpen record (log_type 0x01)."""


@dataclass(frozen=True)
class KeypadWrongCode(_KeypadCode):
    """A KeypadWrongCode record (log_type 0x02)."""


@dataclass(frozen=True)
class KeypadClose(_KeypadCode):
    """A KeypadClose record (log_type 0x08)."""


@dataclass(frozen=True)
class CodeAdded(LogRecord):
    """A CodeAdded record (log_type 0x03)."""

    time_stamp: int
    slot: int
    code: str

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            slot=raw[9],
            code=_code(raw[10:16]),
        )


@dataclass(frozen=True)
class KeypadInfo(LogRecord):
    """A KeypadInfo record (log_type 0x04)."""

    time_stamp: int
    battery_level: int
    stats: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            battery_level=raw[9],
            stats=_u32(raw, 10),
        )


@dataclass(frozen=True)
class CodeDeleted(LogRecord):
    """A CodeDeleted record (log_type 0x05)."""

    time_stamp: int
    slot: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5), slot=raw[9])


@dataclass(frozen=True)
class _TimeStamp(LogRecord):
    """The layout of the AllCodesDeleted and FactoryReset records."""

    time_stamp: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5))


@dataclass(frozen=True)
class AllCodesDeleted(_TimeStamp):
    """An AllCodesDeleted record (log_type 0x06)."""


@dataclass(frozen=True)
class FactoryReset(_TimeStamp):
    """A FactoryReset record (log_type 0x09)."""


@dataclass(frozen=True)
class CodeOperation(LogRecord):
    """A CodeOperation record (log_type 0x07)."""

    time_stamp: int
    current_lock_state: LockStatus
    current_door_state: DoorStatus
    slot: int
    error: int
    battery_level: int
    battery_level_percent: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            current_lock_state=_lock_state(raw[9] & 0x0F),
            current_door_state=_door_state(raw[9] >> 4),
            slot=_u16(raw, 14),
            error=raw[10],
            battery_level=_u16(raw, 12),
            battery_level_percent=raw[11],
        )


@dataclass(frozen=True)
class CodeAddedv2(LogRecord):
    """A CodeAddedv2 record (log_type 0x0A)."""

    time_stamp: int
    slot: int
    code: str

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls._decode(raw, "016X")

    @classmethod
    def decode_decimal(cls, raw: bytes) -> Self:
        """Build the record with the code in decimal, as August's app prints it."""
        return cls._decode(raw, "d")

    @classmethod
    def _decode(cls, raw: bytes, code_format: str) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            slot=_u16(raw, 9),
            code=format(_bytes_to_int(raw[11:16]), code_format),
        )


@dataclass(frozen=True)
class CodeDeletedv2(LogRecord):
    """A CodeDeletedv2 record (log_type 0x0B)."""

    time_stamp: int
    slot: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5), slot=_u16(raw, 9))


@dataclass(frozen=True)
class _Credential(LogRecord):
    """The layout of the three credential records, log_type 0x0C to 0x0E."""

    time_stamp: int
    credential_type: LockCred | int
    slot: int
    credential_error: int | None
    current_lock_state: LockStatus | None
    current_door_state: DoorStatus | None
    battery_level: int
    battery_level_percent: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        # A high nibble of 0xF makes byte 12 an error instead of the two states.
        is_error = raw[12] >> 4 == 0x0F
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            credential_type=_named(LockCred, raw[9]),
            slot=_u16(raw, 10),
            credential_error=raw[12] if is_error else None,
            current_lock_state=None if is_error else _lock_state(raw[12] & 0x0F),
            current_door_state=None if is_error else _door_state(raw[12] >> 4),
            battery_level=_u16(raw, 14),
            battery_level_percent=raw[13],
        )


@dataclass(frozen=True)
class CredentialAdded(_Credential):
    """A CredentialAdded record (log_type 0x0C).

    August's app names it ExternalCredentialAdded.
    """


@dataclass(frozen=True)
class CredentialDeleted(_Credential):
    """A CredentialDeleted record (log_type 0x0D).

    August's app names it ExternalCredentialDeleted.
    """


@dataclass(frozen=True)
class CredentialOperation(_Credential):
    """A CredentialOperation record (log_type 0x0E)."""


@dataclass(frozen=True)
class Tamper(LogRecord):
    """A Tamper record (log_type 0x0F)."""

    time_stamp: int
    tamper_type: TamperType | int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            tamper_type=_named(TamperType, _u16(raw, 9)),
        )


@dataclass(frozen=True)
class BCMTest(LogRecord):
    """A BCMTest record (log_type 0x10)."""

    time_stamp: int
    error: int
    seconds: int
    adv_counter: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            error=raw[9],
            seconds=_u16(raw, 10),
            adv_counter=_u32(raw, 12),
        )


@dataclass(frozen=True)
class HomekeyOperation(LogRecord):
    """A HomekeyOperation record (log_type 0x12)."""

    time_stamp: int
    current_lock_state: LockStatus
    battery_level_decivolt: int
    key_id: int
    error: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            current_lock_state=_lock_state(raw[9] & 0x0F),
            battery_level_decivolt=raw[11],
            key_id=_u32(raw, 12),
            error=raw[10],
        )


@dataclass(frozen=True)
class DPSInfo(LogRecord):
    """A DPSInfo record (log_type 0x13)."""

    time_stamp: int
    battery_level_percent: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5), battery_level_percent=raw[9])


@dataclass(frozen=True)
class BatteryInfo(LogRecord):
    """A BatteryInfo record (log_type 0x15 in Yale's numbering)."""

    time_stamp: int
    battery_level_millivolt: int
    temperature: int
    state_of_charge: int
    state_of_health: int
    battery_index: int
    error: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            battery_level_millivolt=_u16(raw, 9),
            temperature=int.from_bytes(raw[11:12], "little", signed=True),
            state_of_charge=raw[12],
            state_of_health=raw[13],
            battery_index=raw[14],
            error=raw[15],
        )


@dataclass(frozen=True)
class LocalEventNotification(LogRecord):
    """A LocalEventNotification record (log_type 0x16 in Yale's numbering)."""

    time_stamp: int
    notification: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(raw, raw[4], time_stamp=_u32(raw, 5), notification=raw[9])


@dataclass(frozen=True)
class _Message(LogRecord):
    """The layout of the ProductMessage and MatterMessage records."""

    time_stamp: int
    msg_type: int
    data: tuple[int, ...]

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            msg_type=_u16(raw, 9),
            data=tuple(raw[11:16]),
        )


@dataclass(frozen=True)
class ProductMessage(_Message):
    """A ProductMessage record (log_type 0x17 in Yale's numbering)."""


@dataclass(frozen=True)
class MatterMessage(_Message):
    """A MatterMessage record (log_type 0x18 in Yale's numbering)."""


@dataclass(frozen=True)
class DoorStateChanged(LogRecord):
    """A DoorStateChanged record (log_type 0x20)."""

    time_stamp: int
    current_door_state: DoorStatus

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            current_door_state=_door_state(raw[9]),
        )


@dataclass(frozen=True)
class DialogReboot(LogRecord):
    """A DialogReboot record (log_type 0x40)."""

    time_stamp: int
    lock_state: LockStatus
    door_state: DoorStatus
    num_ble_conn: int
    reboot_reason: int
    task_id: int
    pc: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            lock_state=_lock_state(raw[9] & 0x0F),
            door_state=_door_state(raw[9] >> 4),
            num_ble_conn=raw[10] >> 4,
            reboot_reason=raw[10] & 0x0F,
            task_id=raw[11],
            pc=_u32(raw, 12),
        )


@dataclass(frozen=True)
class GetLogError(LogRecord):
    """A GetLogError record (log_type 0x80)."""

    error: int
    key_slot: int
    log_count: int
    dirty_bit: int
    dirty_bit_set_time: int
    dirty_bit_cleared_time: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        bit_fields = _u16(raw, 5)
        return cls(
            raw,
            raw[4],
            error=bit_fields & 0x07,
            key_slot=raw[7],
            log_count=bit_fields >> 4,
            dirty_bit=int(bool(bit_fields & GET_LOG_ERROR_DIRTY_BIT)),
            dirty_bit_set_time=_u32(raw, 8),
            dirty_bit_cleared_time=_u32(raw, 12),
        )


@dataclass(frozen=True)
class LockOperatingMode(LogRecord):
    """A LockOperatingMode record (log_type 0x14 in August's numbering)."""

    time_stamp: int
    lock_op_mode: OperatingMode | int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            lock_op_mode=_named(OperatingMode, raw[9]),
        )


@dataclass(frozen=True)
class LockDeadboltJam(LogRecord):
    """A LockDeadboltJam record (log_type 0x16 in August's numbering)."""

    time_stamp: int
    battery_level: int
    jam_state: JamState | int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            battery_level=_u16(raw, 9),
            jam_state=_named(JamState, raw[11]),
        )


@dataclass(frozen=True)
class InvalidCredentialEntry(LogRecord):
    """An InvalidCredentialEntry record (log_type 0x17 in August's numbering)."""

    time_stamp: int
    cred_type: CredType | int
    battery_level: int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            cred_type=_named(CredType, _u16(raw, 9)),
            battery_level=_u16(raw, 11),
        )


@dataclass(frozen=True)
class MatterEvent(LogRecord):
    """A MatterEvent record (log_type 0x18 in August's numbering)."""

    time_stamp: int
    event_type: MatterEventType | int
    fabric_index: int | None
    fabric_vendor_id: int | None
    # registration_status is None unless event_type is MatterEventType.REGISTRATION.
    # MatterRegistrationStatus.OK is 0, so a truth test cannot tell it from None.
    registration_status: MatterRegistrationStatus | int | None

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        event_type = raw[9]
        has_fabric = event_type in (
            MatterEventType.FABRIC_ADDED,
            MatterEventType.FABRIC_REMOVED,
        )
        is_registration = event_type == MatterEventType.REGISTRATION
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            event_type=_named(MatterEventType, event_type),
            fabric_index=raw[10] if has_fabric else None,
            fabric_vendor_id=_u16(raw, 11) if has_fabric else None,
            registration_status=(
                _named(MatterRegistrationStatus, raw[10]) if is_registration else None
            ),
        )


@dataclass(frozen=True)
class LockOperationBlame(LogRecord):
    """A LockOperationBlame record (log_type 0x19 in August's numbering)."""

    time_stamp: int
    op_type: LockOperationType | int
    op_source: AugustLockOperationSource | int
    op_lock_cred_types: LockOperationCredType | None
    key_slot: int | None
    node_id: int | None
    user_slot: int | None
    fabric_index: int | None
    op_status: LockOperationStatus | int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        op_source = _named(AugustLockOperationSource, raw[10])
        key_slot: int | None = None
        node_id: int | None = None
        user_slot: int | None = None
        fabric_index: int | None = None
        if op_source == AugustLockOperationSource.ZWAVE:
            key_slot = _u16(raw, 12)
            node_id = raw[14]
        elif op_source != AugustLockOperationSource.MANUAL:
            user_slot = _u16(raw, 12)
            if op_source == AugustLockOperationSource.MATTER:
                fabric_index = raw[14]
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            op_type=_named(LockOperationType, raw[9]),
            op_source=op_source,
            # Bit 0 of byte 11 names no credential type.
            op_lock_cred_types=LockOperationCredType(raw[11] & 0xFE) or None,
            key_slot=key_slot,
            node_id=node_id,
            user_slot=user_slot,
            fabric_index=fabric_index,
            op_status=_named(LockOperationStatus, raw[15]),
        )


@dataclass(frozen=True)
class OutOfScheduleCredential(LogRecord):
    """An OutOfScheduleCredential record (log_type 0x1D in August's numbering)."""

    time_stamp: int
    cred_type: CredType | int
    battery_level: int
    violation_type: OutOfScheduleType | int

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            cred_type=_named(CredType, _u16(raw, 9)),
            battery_level=_u16(raw, 11),
            violation_type=_named(OutOfScheduleType, raw[13]),
        )


@dataclass(frozen=True)
class BulkOperation(LogRecord):
    """A BulkOperation record (log_type 0x1E in August's numbering)."""

    time_stamp: int
    bulk_op_type: BulkOperationType | int
    fabric_index: int | None
    lock_cred_type: LockOperationCredType | int | None
    number_of_removed_users: int | None

    @classmethod
    def decode(cls, raw: bytes) -> Self:
        bulk_op_type = raw[9]
        fabric_index: int | None = None
        lock_cred_type: LockOperationCredType | int | None = None
        number_of_removed_users: int | None = None
        if bulk_op_type == BulkOperationType.USERS_CLEAR:
            fabric_index = raw[10]
            number_of_removed_users = _u16(raw, 11)
        elif bulk_op_type == BulkOperationType.CREDS_CLEAR:
            lock_cred_type = _LOCK_OPERATION_CRED_TYPES.get(raw[10], raw[10])
            fabric_index = raw[11]
            number_of_removed_users = _u16(raw, 12)
        if fabric_index == _NO_FABRIC_INDEX:
            fabric_index = None
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            bulk_op_type=_named(BulkOperationType, bulk_op_type),
            fabric_index=fabric_index,
            lock_cred_type=lock_cred_type,
            number_of_removed_users=number_of_removed_users,
        )


RecordDecoder = Callable[[bytes], LogRecord]

YALE_LOG_RECORDS: Mapping[int, RecordDecoder] = {
    LogType.LOCK_OPERATION: LockOperation.decode,
    LogType.KEYPAD_OPEN: KeypadOpen.decode,
    LogType.KEYPAD_WRONG_CODE: KeypadWrongCode.decode,
    LogType.CODE_ADDED: CodeAdded.decode,
    LogType.KEYPAD_INFO: KeypadInfo.decode,
    LogType.CODE_DELETED: CodeDeleted.decode,
    LogType.ALL_CODES_DELETED: AllCodesDeleted.decode,
    LogType.CODE_OPERATION: CodeOperation.decode,
    LogType.KEYPAD_CLOSE: KeypadClose.decode,
    LogType.FACTORY_RESET: FactoryReset.decode,
    LogType.CODE_ADDED_V2: CodeAddedv2.decode,
    LogType.CODE_DELETED_V2: CodeDeletedv2.decode,
    LogType.CREDENTIAL_ADDED: CredentialAdded.decode,
    LogType.CREDENTIAL_DELETED: CredentialDeleted.decode,
    LogType.CREDENTIAL_OPERATION: CredentialOperation.decode,
    LogType.TAMPER: Tamper.decode,
    LogType.BCM_TEST: BCMTest.decode,
    LogType.HOMEKEY_OPERATION: HomekeyOperation.decode,
    LogType.DPS_INFO: DPSInfo.decode,
    LogType.BATTERY_INFO: BatteryInfo.decode,
    LogType.LOCAL_EVENT_NOTIFICATION: LocalEventNotification.decode,
    LogType.PRODUCT_MESSAGE: ProductMessage.decode,
    LogType.MATTER_MESSAGE: MatterMessage.decode,
    LogType.DOOR_STATE_CHANGED: DoorStateChanged.decode,
    LogType.DIALOG_REBOOT: DialogReboot.decode,
    LogType.GET_LOG_ERROR: GetLogError.decode,
}

AUGUST_LOG_RECORDS: Mapping[int, RecordDecoder] = {
    AugustLogType.LOCK_OPERATION: LockOperation.decode,
    AugustLogType.KEYPAD_OPEN: KeypadOpen.decode,
    AugustLogType.KEYPAD_WRONG_CODE: KeypadWrongCode.decode,
    AugustLogType.CODE_ADDED: CodeAdded.decode,
    AugustLogType.KEYPAD_INFO: KeypadInfo.decode,
    AugustLogType.CODE_DELETED: CodeDeleted.decode,
    AugustLogType.ALL_CODES_DELETED: AllCodesDeleted.decode,
    AugustLogType.CODE_OPERATION: CodeOperation.decode,
    AugustLogType.KEYPAD_CLOSE: KeypadClose.decode,
    AugustLogType.FACTORY_RESET: FactoryReset.decode,
    AugustLogType.CODE_ADDED_V2: CodeAddedv2.decode_decimal,
    AugustLogType.CODE_DELETED_V2: CodeDeletedv2.decode,
    AugustLogType.EXTERNAL_CREDENTIAL_ADDED: CredentialAdded.decode,
    AugustLogType.EXTERNAL_CREDENTIAL_DELETED: CredentialDeleted.decode,
    AugustLogType.CREDENTIAL_OPERATION: CredentialOperation.decode,
    AugustLogType.TAMPER: Tamper.decode,
    AugustLogType.BCM_TEST: BCMTest.decode,
    AugustLogType.HOMEKEY_OPERATION: HomekeyOperation.decode,
    AugustLogType.DPS_INFO: DPSInfo.decode,
    AugustLogType.LOCK_OPERATING_MODE: LockOperatingMode.decode,
    AugustLogType.LOCK_DEADBOLT_JAM: LockDeadboltJam.decode,
    AugustLogType.INVALID_CREDENTIAL_ENTRY: InvalidCredentialEntry.decode,
    AugustLogType.MATTER_EVENT: MatterEvent.decode,
    AugustLogType.LOCK_OPERATION_BLAME: LockOperationBlame.decode,
    AugustLogType.OUT_OF_SCHEDULE_CREDENTIAL: OutOfScheduleCredential.decode,
    AugustLogType.BULK_OPERATION: BulkOperation.decode,
    AugustLogType.DOOR_STATE_CHANGED: DoorStateChanged.decode,
    AugustLogType.DIALOG_REBOOT: DialogReboot.decode,
    AugustLogType.GET_LOG_ERROR: GetLogError.decode,
}

_LOG_RECORDS: Mapping[LogBrand, Mapping[int, RecordDecoder]] = {
    LogBrand.YALE: YALE_LOG_RECORDS,
    LogBrand.AUGUST: AUGUST_LOG_RECORDS,
}


def parse_log_event_packet(
    response_packet: bytes, brand: LogBrand = LogBrand.YALE
) -> LogRecord:
    """Decode a GetLog reply into its record, by the log_type at byte 4.

    brand selects Yale's or August's numbering. A log_type with no decoder in
    that numbering gives a plain LogRecord.
    """
    response_packet = bytes(response_packet)
    if (
        len(response_packet) != RESPONSE_FRAME_LEN
        or response_packet[0] != 0xBB
        or response_packet[1] != Commands.GETLOG
    ):
        raise ValueError(
            f"Not an activity log reply: {len(response_packet)} bytes "
            f"starting {response_packet[:2].hex()}"
        )
    records = _LOG_RECORDS[brand]
    return records.get(response_packet[4], LogRecord.decode)(response_packet)
