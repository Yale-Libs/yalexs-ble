from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum
from typing import TypedDict

COMMAND_SERVICE_UUID = "0000fe24-0000-1000-8000-00805f9b34fb"
WRITE_CHARACTERISTIC = "bd4ac611-0b45-11e3-8ffd-0800200c9a66"
READ_CHARACTERISTIC = "bd4ac612-0b45-11e3-8ffd-0800200c9a66"
SECURE_WRITE_CHARACTERISTIC = "bd4ac613-0b45-11e3-8ffd-0800200c9a66"
SECURE_READ_CHARACTERISTIC = "bd4ac614-0b45-11e3-8ffd-0800200c9a66"

APPLE_MFR_ID = 76
YALE_MFR_ID = 465
HAP_FIRST_BYTE = 0x06
HAP_ENCRYPTED_FIRST_BYTE = 0x11


MANUFACTURER_NAME_CHARACTERISTIC = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_NUMBER_CHARACTERISTIC = "00002a24-0000-1000-8000-00805f9b34fb"
SERIAL_NUMBER_CHARACTERISTIC = "00002a25-0000-1000-8000-00805f9b34fb"
FIRMWARE_REVISION_CHARACTERISTIC = "00002a26-0000-1000-8000-00805f9b34fb"

NO_DOOR_SENSE_MODELS = {"ASL-02", "ASL-01"}


class Commands(IntEnum):
    GETSTATUS = 0x02
    WRITESETTING = 0x03
    READSETTING = 0x04
    UNLOCK = 0x0A
    LOCK = 0x0B
    LOCK_ACTIVITY = 0x2D


class OperationError(IntEnum):
    """Operation result reported in byte[15] of an op-response (BB 0A/0B).

    0x00 = success; any non-zero value is a failure (0x1E-0x23 are the
    MECH_* motor faults, i.e. a stalled motor / jam). Non-mechanical
    failures (credential, battery, wifi, calibration) are included for
    logging so a failure report names the actual cause.
    """

    COMM_SUCCESS = 0x00
    PARAM_NOT_PAIRED = 0x01
    PARAM_NOT_READABLE = 0x02
    WRONG_KEY = 0x03
    # Credential-management (keypad / RFID / fingerprint / palm) errors, not
    # mechanical faults.
    KEYCODE_DISABLE = 0x04
    KEYCODE_INVALID_ACCESS = 0x05
    KEYCODE_EXISTING_KEY = 0x06
    KEYCODE_NOSPACE = 0x07
    KEYCODE_TIMEOUT = 0x08
    KEYCODE_DIS_ONETOUCH = 0x09
    RFID_EXISTING_ID = 0x0A
    FINGERPRINT_EXISTING_ID = 0x0B
    PALM_EXISTING_ID = 0x0D
    # Mechanical / motor faults (a stalled motor / jam).
    MECH_TIMEOUT = 0x1E
    MECH_POSITION = 0x1F
    MECH_MOTPOL = 0x20
    MECH_TIMEOUT_CAL = 0x21
    MECH_BACKOFF = 0x22
    MECH_HANDLE_NOT_LIFTED = 0x23
    EMPTY_LOG = 0x28
    READING_LOG = 0x29
    VBAT_LOW = 0x32
    OVERTEMP = 0x33
    MAG_READ = 0x34
    MAG_BAD_DATA = 0x35
    INVALID_OPERATION = 0x37
    NOT_AUTHORIZED = 0x38
    NO_BUFS = 0x39
    INVALID_MSG = 0x3A
    NACK = 0x3B
    UNKNOWN = 0x3C
    UNINITIALIZED = 0x3D
    BUSY = 0x3E
    NOT_FOUND = 0x3F
    SSID_IS_5GHZ = 0x40
    DHCP_FAILURE = 0x41
    DNS_FAILURE = 0x42
    CAL_BAD_EXTENTS = 0x43
    CAL_BAD_ANGLE = 0x44
    KEYCODE_SLOT_IN_USE = 0x45
    EXT_TIMEOUT = 0x46


VALUE_TO_OPERATION_ERROR = {err.value: err for err in OperationError}


class StatusType(IntEnum):
    LOCK_ONLY = 0x02
    DOOR_ONLY = 0x2E
    DOOR_AND_LOCK = 0x2F
    BATTERY = 0x0F


class SettingType(IntEnum):
    AUTOLOCK = 0x28


class LockStatus(Enum):
    UNKNOWN = 0x00
    UNKNOWN_01 = 0x01  # Calibrating
    UNLOCKING = 0x02
    UNLOCKED = 0x03
    LOCKING = 0x04
    LOCKED = 0x05
    UNKNOWN_06 = 0x06  # PolDiscovery
    JAMMED = 0x07  # STATICPOSITION
    # UNLATCHING = 0x09
    # UNLATCHED = 0x0A
    SECUREMODE = 0x0C


VALUE_TO_LOCK_STATUS = {status.value: status for status in LockStatus}


class DoorStatus(Enum):
    UNKNOWN = 0x00  # Init
    CLOSED = 0x01
    AJAR = 0x02
    OPENED = 0x03
    UNKNOWN_04 = 0x04  # Unknown


VALUE_TO_DOOR_STATUS = {status.value: status for status in DoorStatus}


class AutoLockMode(IntEnum):
    # The values are arbitrary; the mode never crosses the wire. It is derived
    # from the stored value's shape on read (see Lock._parse_auto_lock_state)
    # and implied by that shape on write (see Lock.set_auto_lock).
    INSTANT = 0
    TIMER = 1
    OFF = 2


class LockActivityType(Enum):
    LOCK = 0x00
    DOOR = 0x20
    PIN = 0x0E
    NONE = 0x80


@dataclass
class BatteryState:
    voltage: float
    percentage: int


@dataclass
class AutoLockState:
    mode: AutoLockMode
    duration: int


@dataclass
class LockState:
    lock: LockStatus
    door: DoorStatus
    battery: BatteryState | None
    auth: AuthState | None
    auto_lock: AutoLockState | None
    # Hold the previous auto lock state so that it can be restored if auto lock
    # is enabled
    auto_lock_prev: AutoLockState | None


LockStateValue = LockStatus | DoorStatus | BatteryState | AutoLockState


class LockOperationSource(Enum):
    REMOTE = 0x00
    MANUAL = 0x01
    AUTO_LOCK = 0x05
    PIN = 0x0B
    UNKNOWN = 0xFF


VALUE_TO_LOCK_OPERATION_SOURCE = {
    status.value: status for status in LockOperationSource
}


class LockOperationRemoteType(Enum):
    UNKNOWN = 0x00
    BLE = 0x03


VALUE_TO_LOCK_OPERATION_REMOTE_TYPE = {
    status.value: status for status in LockOperationRemoteType
}


@dataclass
class LockActivity:
    timestamp: datetime
    status: LockStatus
    source: LockOperationSource
    remote_type: LockOperationRemoteType | None = None
    slot: int | None = None


@dataclass
class DoorActivity:
    timestamp: datetime
    status: DoorStatus


@dataclass
class AuthState:
    successful: bool


@dataclass
class LockInfo:
    manufacturer: str
    model: str
    serial: str
    firmware: str

    @property
    def door_sense(self) -> bool:
        """Check if the lock has door sense support."""
        return bool(
            self.model
            and not any(
                self.model.startswith(old_model) for old_model in NO_DOOR_SENSE_MODELS
            )
        )


@dataclass
class ConnectionInfo:
    rssi: int


class YaleXSBLEDiscovery(TypedDict):
    """A validated discovery of a Yale XS BLE device."""

    name: str
    address: str
    serial: str
    key: str
    slot: int
