from bleak_retry_connector import close_stale_connections_by_address

from .const import (
    AutoLockMode,
    ConnectionInfo,
    DoorStatus,
    LockInfo,
    LockState,
    LockStatus,
    YaleXSBLEDiscovery,
)
from .lock import Lock
from .push import PushLock
from .session import AuthError, DisconnectedError, YaleXSBLEError
from .util import (
    ValidatedLockConfig,
    local_name_is_unique,
    local_name_to_serial,
    serial_to_local_name,
    unique_id_from_device_adv,
    unique_id_from_local_name_address,
)

__version__ = "3.2.4"

__all__ = [
    "AuthError",
    "AutoLockMode",
    "ConnectionInfo",
    "DisconnectedError",
    "DoorStatus",
    "Lock",
    "LockInfo",
    "LockState",
    "LockStatus",
    "PushLock",
    "ValidatedLockConfig",
    "YaleXSBLEDiscovery",
    "YaleXSBLEError",
    "close_stale_connections_by_address",
    "local_name_is_unique",
    "local_name_to_serial",
    "serial_to_local_name",
    "unique_id_from_device_adv",
    "unique_id_from_local_name_address",
]
