from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
from typing import TypedDict

COMMAND_SERVICE_UUID = "0000fe24-0000-1000-8000-00805f9b34fb"
WRITE_CHARACTERISTIC = "bd4ac611-0b45-11e3-8ffd-0800200c9a66"
READ_CHARACTERISTIC = "bd4ac612-0b45-11e3-8ffd-0800200c9a66"
SECURE_WRITE_CHARACTERISTIC = "bd4ac613-0b45-11e3-8ffd-0800200c9a66"
SECURE_READ_CHARACTERISTIC = "bd4ac614-0b45-11e3-8ffd-0800200c9a66"

APPLE_MFR_ID = 76
YALE_MFR_ID = 465
ASSA_ABLOY_MFR_ID = 302
# Company IDs a lock's manufacturer data is advertised under, read in this order.
YALE_MFR_IDS = (YALE_MFR_ID, ASSA_ABLOY_MFR_ID)
HAP_FIRST_BYTE = 0x06
HAP_ENCRYPTED_FIRST_BYTE = 0x11

# Every frame on the link, sent or received, is 18 bytes: a 16-byte encrypted
# block plus two trailing plain bytes, and the checksums run over all 18. The
# command builders, the checksum helpers, and the notify admission gate all
# share this length.
RESPONSE_FRAME_LEN = 0x12

# When True, a log line that prints a decrypted or plaintext frame masks its
# keypad code and the checksum that gives away the sum of the code's bytes.
MASK_SENSITIVE = False


MANUFACTURER_NAME_CHARACTERISTIC = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_NUMBER_CHARACTERISTIC = "00002a24-0000-1000-8000-00805f9b34fb"
SERIAL_NUMBER_CHARACTERISTIC = "00002a25-0000-1000-8000-00805f9b34fb"
FIRMWARE_REVISION_CHARACTERISTIC = "00002a26-0000-1000-8000-00805f9b34fb"

# The Model Number characteristic carries regional and firmware suffixes
# such as "SL-103-EU", so the tables match an entry or a suffixed variant.
NO_DOOR_SENSE_MODELS = {"ASL-02", "ASL-01"}

# With BATTERY_TIMEOUT_COOLDOWN it may be possible to remove these exclusions
NO_BATTERY_SUPPORT_MODELS = {
    "SL-103",  # Linus L2
    "CERES",  # Smart code handle
    "Yale Linus L2",  # Linus L2 Nordic
}


class Commands(IntEnum):
    GETSTATUS = 0x02
    WRITESETTING = 0x03
    READSETTING = 0x04
    UNLOCK = 0x0A
    LOCK = 0x0B
    KEYCODE_SET = 0x27
    KEYCODE_CLEAR = 0x28
    KEYCODE_ACCESS = 0x2B
    KEYCODE_COMMIT = 0x2C
    GETLOG = 0x2D
    KEYCODE_GET = 0x39


# The log_type at byte 4 of the GETLOG reply that ends the log, the vendor's
# GetLogError record.
LOG_TYPE_GET_LOG_ERROR = 0x80


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

# The results of a motor stall or a mechanism out of position.
MECHANICAL_OPERATION_ERRORS = frozenset(
    {
        OperationError.MECH_TIMEOUT,
        OperationError.MECH_POSITION,
        OperationError.MECH_MOTPOL,
        OperationError.MECH_TIMEOUT_CAL,
        OperationError.MECH_BACKOFF,
        OperationError.MECH_HANDLE_NOT_LIFTED,
    }
)


class StatusType(IntEnum):
    LOCK_ONLY = 0x02
    DOOR_ONLY = 0x2E
    DOOR_AND_LOCK = 0x2F
    BATTERY = 0x0F
    # The activity log's unread count: how many records no client has read
    # yet. The read removes no record.
    LOCK_EVENTS_UNREAD = 0x09


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
    UNLATCHING = 0x09
    UNLATCHED = 0x0A
    INTEGRATED_OTHER_SYSTEM = 0x0B
    SECUREMODE = 0x0C
    MASTER_CODE_NOT_SET = 0x0D
    DEFAULT_MASTER_CODE_SET = 0x0E
    LOCK_NOT_HANDED = 0x0F
    SHABBAT_MODE = 0x10
    INTERNAL_FORCED_LOCKED = 0x11


VALUE_TO_LOCK_STATUS = {status.value: status for status in LockStatus}

# Statuses reported during calibration (0x01) and polarity discovery (0x06),
# setup conditions that end at the lock by hand.
SETUP_CONDITION_STATUSES = frozenset({LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06})

# Statuses that need a person at the lock; exported for consumers.
MANUAL_INTERVENTION_STATUSES = SETUP_CONDITION_STATUSES | {LockStatus.JAMMED}


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


@dataclass
class BatteryState:
    voltage: float
    percentage: int


@dataclass
class ActivityLogDepth:
    """The activity log's unread count as the lock reports it."""

    unread: int

    @property
    def is_depth(self) -> bool:
        """True when the value is a count; a negative value is not one."""
        return self.unread >= 0


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
    # Status of the secure lock: LOCKED when the lock is in securemode, so
    # SECUREMODE itself never appears here.
    secure: LockStatus = LockStatus.UNKNOWN


LockStateValue = LockStatus | DoorStatus | BatteryState | AutoLockState


class OpCode(IntEnum):
    """The opCode at byte 5 of a type 0x00 activity record."""

    # Based on the vendor's term. Locks have been observed to record a lock or unlock
    # commanded over BLE with this value.
    DOOR_STATE_CHANGED = 0x00
    DOOR_MANUAL_STATE_CHANGE = 0x01
    KNOCK_RECEIVED = 0x02
    BLUETOOTH_RECEIVED = 0x03
    ANOMALY = 0x04
    AUTO_RELOCK = 0x05
    AUTO_RETRY = 0x06
    OVER_TEMP = 0x07
    BURN_IN = 0x08
    HOME_KIT = 0x09
    Z_WAVE = 0x0A
    ONE_TOUCH = 0x0B
    INSIDE_BUTTON = 0x0C
    DOOR_SENSE = 0x0E
    LOW_BAT_AUTO_UNLOCK = 0x0F
    ESCAPE_UNLOCK = 0x10
    UWB = 0x11
    EXTERNAL_MODULE_OPERATION = 0x12
    SHABBAT_MODE = 0x13
    SCHEDULED_OPERATION = 0x14
    MATTER = 0x15
    DOUBLE_PRESS = 0x17
    ALIRO = 0x18
    UNDEFINED = 0xFF


class LogBrand(Enum):
    """The brand whose numbering and names an activity log record follows."""

    YALE = "yale"
    AUGUST = "august"


class LogType(IntEnum):
    """The log_type at byte 4 of a GetLog reply, in Yale's numbering."""

    LOCK_OPERATION = 0x00
    KEYPAD_OPEN = 0x01
    KEYPAD_WRONG_CODE = 0x02
    CODE_ADDED = 0x03
    KEYPAD_INFO = 0x04
    CODE_DELETED = 0x05
    ALL_CODES_DELETED = 0x06
    CODE_OPERATION = 0x07
    KEYPAD_CLOSE = 0x08
    FACTORY_RESET = 0x09
    CODE_ADDED_V2 = 0x0A
    CODE_DELETED_V2 = 0x0B
    CREDENTIAL_ADDED = 0x0C
    CREDENTIAL_DELETED = 0x0D
    CREDENTIAL_OPERATION = 0x0E
    TAMPER = 0x0F
    BCM_TEST = 0x10
    ACCESSORY_EVENT = 0x11
    HOMEKEY_OPERATION = 0x12
    DPS_INFO = 0x13
    BATTERY_INFO = 0x15
    LOCAL_EVENT_NOTIFICATION = 0x16
    PRODUCT_MESSAGE = 0x17
    MATTER_MESSAGE = 0x18
    DOOR_STATE_CHANGED = 0x20
    DIALOG_REBOOT = 0x40
    GET_LOG_ERROR = LOG_TYPE_GET_LOG_ERROR


class AugustLogType(IntEnum):
    """The log_type at byte 4 of a GetLog reply, in August's numbering."""

    LOCK_OPERATION = 0x00
    KEYPAD_OPEN = 0x01
    KEYPAD_WRONG_CODE = 0x02
    CODE_ADDED = 0x03
    KEYPAD_INFO = 0x04
    CODE_DELETED = 0x05
    ALL_CODES_DELETED = 0x06
    CODE_OPERATION = 0x07
    KEYPAD_CLOSE = 0x08
    FACTORY_RESET = 0x09
    CODE_ADDED_V2 = 0x0A
    CODE_DELETED_V2 = 0x0B
    EXTERNAL_CREDENTIAL_ADDED = 0x0C
    EXTERNAL_CREDENTIAL_DELETED = 0x0D
    CREDENTIAL_OPERATION = 0x0E
    TAMPER = 0x0F
    BCM_TEST = 0x10
    ACCESSORY_EVENT = 0x11
    HOMEKEY_OPERATION = 0x12
    DPS_INFO = 0x13
    LOCK_OPERATING_MODE = 0x14
    WIFI_INFO = 0x15
    LOCK_DEADBOLT_JAM = 0x16
    INVALID_CREDENTIAL_ENTRY = 0x17
    MATTER_EVENT = 0x18
    LOCK_OPERATION_BLAME = 0x19
    OUT_OF_SCHEDULE_CREDENTIAL = 0x1D
    BULK_OPERATION = 0x1E
    DOOR_STATE_CHANGED = 0x20
    DIALOG_REBOOT = 0x40
    GET_LOG_ERROR = LOG_TYPE_GET_LOG_ERROR


class LockCred(IntEnum):
    """The credential type of a credential record."""

    PIN = 0x00
    RF = 0x01
    FINGER = 0x02
    FACE = 0x04
    INSIDE_DOOR_PIN = 0x05
    PALM = 0x06
    ALIRO = 0x08
    ALL = 0xFF


class TamperType(IntEnum):
    """The tamper type of a Tamper record."""

    CLEAR = 0x00
    TOO_MANY_AUTH_ERRORS = 0x01
    FRONT_REMOVED = 0x02
    DOOR_FORCED_OPEN = 0x03
    HIGH_TEMPERATURE = 0x04
    TOO_MANY_AUTH_ERRORS_PIN = 0x05
    TOO_MANY_AUTH_ERRORS_PIN_CLEARED = 0x06
    TOO_MANY_AUTH_ERRORS_RFID = 0x07
    TOO_MANY_AUTH_ERRORS_RFID_CLEARED = 0x08
    TOO_MANY_AUTH_ERRORS_FINGERPRINT = 0x09
    TOO_MANY_AUTH_ERRORS_FINGERPRINT_CLEARED = 0x0A
    TOO_MANY_AUTH_ERRORS_PALM = 0x0B
    TOO_MANY_AUTH_ERRORS_PALM_CLEARED = 0x0C
    TOO_MANY_AUTH_ERRORS_FACE_SCAN = 0x0D
    TOO_MANY_AUTH_ERRORS_FACE_SCAN_CLEARED = 0x0E


class OperatingMode(IntEnum):
    """The operating mode of August's LockOperatingMode record."""

    NORMAL = 0x00
    VACATION = 0x01
    PRIVACY = 0x02
    PASSAGE = 0x03


class JamState(IntEnum):
    """The jam state of August's LockDeadboltJam record."""

    OCCURS_DURING_LOCKING = 0x00
    OCCURS_DURING_UNLOCKING = 0x01


class CredType(IntEnum):
    """The cred_type of August's InvalidCredentialEntry and OutOfScheduleCredential."""

    ENTRY_KEYPAD_PINCODE = 0x0000
    ENTRY_KEYPAD_FINGERPRINT = 0x0002
    ENTRY_AUX_KEYPAD_PINCODE = 0x0100
    ENTRY_AUX_KEYPAD_FINGERPRINT = 0x0102
    ENTRY_APPLE_HOMEKEYS_NFC = 0x1001
    ENTRY_SAMSUNG_WALLET_NFC = 0x1101
    ENTRY_GOOGLE_WALLET_NFC = 0x1201
    ENTRY_RFID_CARD_MIFARE_DESFIRE_EV1_NFC = 0x2101
    ENTRY_RFID_CARD_MIFARE_DESFIRE_EV2_NFC = 0x2201
    ENTRY_RFID_CARD_MIFARE_DESFIRE_EV3_NFC = 0x2301
    ENTRY_MATTER_FABRIC_MATTER = 0x3003
    ENTRY_UNKNOWN = 0xFFFF


class OutOfScheduleType(IntEnum):
    """The violation type of August's OutOfScheduleCredential record."""

    PREMATURE = 0x00
    EXPIRED = 0x01


class MatterEventType(IntEnum):
    """The event type of August's MatterEvent record."""

    REGISTRATION = 0x00
    FABRIC_ADDED = 0x01
    FABRIC_REMOVED = 0x02


class MatterRegistrationStatus(IntEnum):
    """The registration status of August's MatterEvent record."""

    OK = 0x00
    FAIL = 0x01
    UNKNOWN = 0xFF


class LockOperationType(IntEnum):
    """The operation type of August's LockOperationBlame record."""

    LOCK = 0x00
    UNLOCK = 0x01


class AugustLockOperationSource(IntEnum):
    """The operation source of August's LockOperationBlame record."""

    BLE = 0x01
    KEYPAD = 0x02
    LOCK_BUTTON = 0x03
    AUTORELOCK = 0x04
    MANUAL = 0x05
    MATTER = 0x06
    ZWAVE = 0x07


class LockOperationCredType(IntFlag):
    """The credential types of August's LockOperationBlame and BulkOperation records."""

    PIN = 0x02
    RF_ID = 0x04
    FINGERPRINT = 0x08
    RF_ID_HID = 0x10
    FACE = 0x20
    INSIDE_DOOR_PIN = 0x40
    BLE = 0x80


class LockOperationStatus(IntEnum):
    """The operation status of August's LockOperationBlame record."""

    SUCCESS = 0x00
    FAIL = 0x01
    LOW_BATTERY = 0x02
    JAMMED = 0x03
    NON_CALIBRATED = 0x04
    TIMEOUT = 0x05
    INVALID_CRED = 0x06
    DISABLED_USER = 0x07
    OUTSIDE_SCHEDULE = 0x08
    NON_ACCESS = 0x09


class BulkOperationType(IntEnum):
    """The operation type of August's BulkOperation record."""

    USERS_CLEAR = 0x01
    CREDS_CLEAR = 0x02


# Each table maps a value to the vendor's name for it; a value the vendor
# does not name has no entry. A suffix names the brand whose app uses the
# names.
LOG_TYPE_NAMES_YALE: dict[int, str] = {
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
}

LOG_TYPE_NAMES_AUGUST: dict[int, str] = {
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
    0x0C: "ExternalCredentialAdded",
    0x0D: "ExternalCredentialDeleted",
    0x0E: "CredentialOperation",
    0x0F: "Tamper",
    0x10: "BCMTest",
    0x11: "AccessoryEvent",
    0x12: "HomekeyOperation",
    0x13: "DPSInfo",
    0x14: "LockOperatingMode",
    0x15: "WifiInfo",
    0x16: "LockDeadboltJam",
    0x17: "InvalidCredentialEntry",
    0x18: "MatterEvent",
    0x19: "LockOperationBlame",
    0x1D: "OutOfScheduleCredential",
    0x1E: "BulkOperation",
    0x20: "DoorStateChanged",
    0x40: "DialogReboot",
    0x80: "GetLogError",
}

OP_CODE_NAMES_YALE: dict[int, str] = {
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
}

OP_CODE_NAMES_AUGUST: dict[int, str] = {
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
    0xFF: "Undefined",
}

LOCK_STATE_NAMES_YALE: dict[int, str] = {
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
}

LOCK_STATE_NAMES_AUGUST: dict[int, str] = {
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
    0x10: "Shabbat Mode",
}

CREDENTIAL_TYPE_NAMES_YALE: dict[int, str] = {
    0x00: "pin",
    0x01: "rf",
    0x02: "finger",
    0x04: "face",
    0x05: "inside_door_pin",
    0x06: "palm",
    0x08: "aliro",
    0xFF: "All",
}

CREDENTIAL_TYPE_NAMES_AUGUST: dict[int, str] = {
    0x00: "pin",
    0x01: "rf",
    0x02: "finger",
    0x04: "face",
    0xFF: "All",
}

DOOR_STATE_NAMES: dict[int, str] = {
    0x00: "Init",
    0x01: "Closed",
    0x02: "Ajar",
    0x03: "Open",
    0x04: "Unknown",
}

TAMPER_TYPE_NAMES_YALE: dict[int, str] = {
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
}

TAMPER_TYPE_NAMES_AUGUST: dict[int, str] = {
    0x00: "TamperClear",
    0x01: "TamperTooManyAuthErrors",
    0x02: "TamperFrontRemoved",
    0x03: "TamperDoorForcedOpen",
    0x04: "TamperHighTemperature",
}

# Only August's app has these names, so their table names have no brand suffix.
OPERATING_MODE_NAMES: dict[int, str] = {
    0x00: "OperatingModeNormal",
    0x01: "OperatingModeVacation",
    0x02: "OperatingModePrivacy",
    0x03: "OperatingModePassage",
}

JAM_STATE_NAMES: dict[int, str] = {
    0x00: "JamOccursDuringLocking",
    0x01: "JamOccursDuringUnlocking",
}

CRED_TYPE_NAMES: dict[int, str] = {
    0x0000: "CredEntryKeypadPincode",
    0x0002: "CredEntryKeypadFingerprint",
    0x0100: "CredEntryAuxKeypadPincode",
    0x0102: "CredEntryAuxKeypadFingerprint",
    0x1001: "CredEntryAppleHomekeysNFC",
    0x1101: "CredEntrySamsungWalletNFC",
    0x1201: "CredEntryGoogleWalletNFC",
    0x2101: "CredEntryRFIDCardMIFAREDESFireEV1NFC",
    0x2201: "CredEntryRFIDCardMIFAREDESFireEV2NFC",
    0x2301: "CredEntryRFIDCardMIFAREDESFireEV3NFC",
    0x3003: "CredEntryMatterFabricMatter",
    0xFFFF: "CredEntryUnknown",
}

OUT_OF_SCHEDULE_TYPE_NAMES: dict[int, str] = {
    0x00: "OutOfSchedulePremature",
    0x01: "OutOfScheduleExpired",
}

MATTER_EVENT_NAMES: dict[int, str] = {
    0x00: "MatterEventRegistration",
    0x01: "MatterEventFabricAdded",
    0x02: "MatterEventFabricRemoved",
}

MATTER_REGISTRATION_STATUS_NAMES: dict[int, str] = {
    0x00: "OK",
    0x01: "FAIL",
    0xFF: "UNKNOWN",
}

LOCK_OPERATION_TYPE_NAMES: dict[int, str] = {
    0x00: "Lock",
    0x01: "Unlock",
}

LOCK_OPERATION_SOURCE_NAMES: dict[int, str] = {
    0x01: "Ble",
    0x02: "Keypad",
    0x03: "Lock_Button",
    0x04: "Autorelock",
    0x05: "Manual",
    0x06: "Matter",
    0x07: "Zwave",
}

LOCK_OPERATION_CRED_TYPE_NAMES: dict[int, str] = {
    0x02: "Pin",
    0x04: "RfId",
    0x08: "Fingerprint",
    0x10: "RfId_Hid",
    0x20: "Face",
    0x40: "InsideDoorPin",
    0x80: "Ble",
}

LOCK_OPERATION_STATUS_NAMES: dict[int, str] = {
    0x00: "Success",
    0x01: "Fail",
    0x02: "Low_Battery",
    0x03: "Jammed",
    0x04: "Non_Calibrated",
    0x05: "Timeout",
    0x06: "Invalid_cred",
    0x07: "Disabled_user",
    0x08: "Outside_Schedule",
    0x09: "Non_Access",
}

BULK_OPERATION_TYPE_NAMES: dict[int, str] = {
    0x01: "UsersClear",
    0x02: "CredsClear",
}


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
        return bool(self.model) and not self._model_in(NO_DOOR_SENSE_MODELS)

    @property
    def battery_reporting(self) -> bool:
        """Check if the lock answers battery status requests."""
        return not self._model_in(NO_BATTERY_SUPPORT_MODELS)

    def _model_in(self, models: set[str]) -> bool:
        """True for a listed model or a suffixed variant of one, not a longer number."""
        return any(
            self.model == model or self.model.startswith((f"{model}-", f"{model} "))
            for model in models
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
