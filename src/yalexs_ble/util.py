from __future__ import annotations

from asyncio import timeout as asyncio_timeout  # noqa: F401
from dataclasses import dataclass

from bleak import BleakError
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

UNIQUE_LOCAL_NAME_LEN = 7


def _simple_checksum(buf: bytes | bytearray) -> int:
    cs = 0
    for i in range(0x12):
        cs = (cs + buf[i]) & 0xFF

    return (-cs) & 0xFF


def _bytes_to_int(buffer: bytes | bytearray) -> int:
    """Convert a byte buffer to an integer."""
    return int.from_bytes(buffer, byteorder="little", signed=False)


def _int_to_bytes(value: int, length: int) -> bytes:
    """Convert an integer to a byte buffer."""
    if length < 1 or length > 8:
        raise ValueError("Length must be between 1 and 8 bytes.")
    return value.to_bytes(length, byteorder="little", signed=False)


def _security_checksum(buffer: bytes | bytearray) -> int:
    val1 = _bytes_to_int(buffer[0x00:0x04])
    val2 = _bytes_to_int(buffer[0x04:0x08])
    val3 = _bytes_to_int(buffer[0x08:0x12])

    return (0 - (val1 + val2 + val3)) & 0xFFFFFFFF


def _copy(dest: bytearray, src: bytes, destLocation: int = 0) -> None:
    dest[destLocation : (destLocation + len(src))] = src


def serial_to_local_name(serial: str) -> str:
    """Convert a serial to a local name."""
    return f"{serial[0:2]}{serial[-5:]}"


def local_name_to_serial(serial: str) -> str:
    """Convert a local name to a serial."""
    return f"{serial[0:2]}XXX{serial[2:]}"


def is_key_error(error: Exception) -> bool:
    """Check if the error likely due to the wrong key."""
    err_str = str(error)
    return bool(
        isinstance(error, BleakError)
        and ("Unlikely Error" in err_str or "error=133" in err_str)
    )


def is_disconnected_error(error: Exception) -> bool:
    """Check if the error is a disconnected error."""
    err_str = str(error)
    return is_key_error(error) or bool(
        isinstance(error, BleakError)
        and (
            "disconnect" in err_str
            or "Connection Rejected Due To Security Reasons" in err_str
        )
    )


@dataclass
class ValidatedLockConfig:
    """A validated lock configuration."""

    name: str
    address: str
    serial: str
    key: str
    slot: int

    @property
    def local_name(self) -> str:
        """Get the local name from the serial."""
        return serial_to_local_name(self.serial)


def unique_id_from_device_adv(
    device: BLEDevice, advertisement: AdvertisementData
) -> str:
    """Get the unique id from the advertisement."""
    return unique_id_from_local_name_address(advertisement.local_name, device.address)


def unique_id_from_local_name_address(local_name: str, address: str) -> str:
    """Get the unique id from the advertisement."""
    return local_name if local_name_is_unique(local_name) else address


def local_name_is_unique(local_name: str | None) -> bool:
    """Check if the local name is unique."""
    return bool(local_name and len(local_name) == UNIQUE_LOCAL_NAME_LEN)
