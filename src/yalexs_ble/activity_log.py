"""Decode the records of the lock's activity log."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self

from .const import (
    RESPONSE_FRAME_LEN,
    VALUE_TO_DOOR_STATUS,
    VALUE_TO_LOCK_STATUS,
    Commands,
    DoorStatus,
    LockCred,
    LockStatus,
    LogType,
    OpCode,
    TamperType,
)
from .util import _bytes_to_int, _named


def _u16(raw: bytes, offset: int) -> int:
    return _bytes_to_int(raw[offset : offset + 2])


def _u32(raw: bytes, offset: int) -> int:
    return _bytes_to_int(raw[offset : offset + 4])


# A state with no member reads UNKNOWN, as the status parsers in lock.py do; raw
# keeps the byte.
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
        return cls(
            raw,
            raw[4],
            time_stamp=_u32(raw, 5),
            slot=_u16(raw, 9),
            code=format(_bytes_to_int(raw[11:16]), "016X"),
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
    """A CredentialAdded record (log_type 0x0C)."""


@dataclass(frozen=True)
class CredentialDeleted(_Credential):
    """A CredentialDeleted record (log_type 0x0D)."""


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
            dirty_bit=bit_fields >> 3 & 0x01,
            dirty_bit_set_time=_u32(raw, 8),
            dirty_bit_cleared_time=_u32(raw, 12),
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


def parse_log_event_packet(response_packet: bytes) -> LogRecord:
    """Decode a GetLog reply into its record, by the log_type at byte 4.

    A log_type with no decoder gives a plain LogRecord.
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
    return YALE_LOG_RECORDS.get(response_packet[4], LogRecord.decode)(response_packet)
