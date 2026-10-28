from __future__ import annotations

import re
from asyncio import timeout as asyncio_timeout  # noqa: F401
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeVar

from bleak import BleakError
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from . import const
from .const import RESPONSE_FRAME_LEN, Commands

UNIQUE_LOCAL_NAME_LEN = 7

_EnumT = TypeVar("_EnumT", bound=IntEnum)


def _require_frame_length(buf: bytes | bytearray) -> None:
    """Refuse a buffer shorter than the frame a checksum runs over.

    The checksum helpers defend their own input rather than lean on the
    notify gate: a slice over a short buffer would silently checksum
    whatever bytes are there.
    """
    if len(buf) < RESPONSE_FRAME_LEN:
        raise ValueError(f"checksum needs {RESPONSE_FRAME_LEN} bytes, got {len(buf)}")


def _simple_checksum(buf: bytes | bytearray) -> int:
    _require_frame_length(buf)
    return (-sum(buf[:RESPONSE_FRAME_LEN])) & 0xFF


def _bytes_to_int(buffer: bytes | bytearray) -> int:
    """Convert a byte buffer to an integer."""
    return int.from_bytes(buffer, byteorder="little", signed=False)


def _bytes_to_signed_int(buffer: bytes | bytearray) -> int:
    """Convert a byte buffer to a signed integer."""
    return int.from_bytes(buffer, byteorder="little", signed=True)


def _named(enum: type[_EnumT], value: int) -> _EnumT | int:
    """Return the member of enum for value, or value when enum has none.

    An unnamed value is returned unchanged, so the decoded record keeps it.
    """
    try:
        return enum(value)
    except ValueError:
        return value


def _int_to_bytes(value: int, length: int) -> bytes:
    """Convert an integer to a byte buffer."""
    if length < 1 or length > 8:
        raise ValueError("Length must be between 1 and 8 bytes.")
    return value.to_bytes(length, byteorder="little", signed=False)


def _security_checksum(buffer: bytes | bytearray) -> int:
    _require_frame_length(buffer)
    val1 = _bytes_to_int(buffer[0x00:0x04])
    val2 = _bytes_to_int(buffer[0x04:0x08])
    val3 = _bytes_to_int(buffer[0x08:RESPONSE_FRAME_LEN])

    return (0 - (val1 + val2 + val3)) & 0xFFFFFFFF


KEYCODE_PIN_BYTES = 7
KEYCODE_MAX_DIGITS = KEYCODE_PIN_BYTES * 2


def encode_keycode_pin(pin: str) -> bytes:
    """Pack a keypad PIN as BCD, first digit in the high nibble, padded with 0xF."""
    if not re.fullmatch(f"[0-9]{{1,{KEYCODE_MAX_DIGITS}}}", pin):
        raise ValueError(f"PIN must be 1-{KEYCODE_MAX_DIGITS} digits (0-9): {pin!r}")
    return bytes.fromhex(pin.ljust(KEYCODE_MAX_DIGITS, "f"))


def decode_keycode_pin(data: bytes | bytearray) -> str | None:
    """Unpack a BCD keypad PIN padded with 0xF; None for an empty slot.

    Raises ValueError for a field that is not digits followed by padding.
    """
    field = data[:KEYCODE_PIN_BYTES].hex()
    pin = field.rstrip("f")
    if pin and not pin.isdigit():
        raise ValueError("Malformed PIN field")
    return pin or None


# The first byte of the keypad code in a GetLog reply, by log_type: KeypadOpen,
# KeypadWrongCode and KeypadClose (0x01, 0x02, 0x08), CodeAdded (0x03), and
# CodeAddedv2 (0x0A). The code runs to byte 15.
_GETLOG_CODE_START = {0x01: 0x09, 0x02: 0x09, 0x08: 0x09, 0x03: 0x0A, 0x0A: 0x0B}

# The commands whose frame and reply have a keypad code field in bytes 4 to 10,
# and whose acknowledgment matches the request's byte 4: KeyCode_Set,
# KeyCode_Clear, KeyCode_Unlock (0x2A, which the library does not send), and
# KeyCode_Commit.
_KEYCODE_COMMANDS = frozenset(
    {Commands.KEYCODE_SET, Commands.KEYCODE_CLEAR, 0x2A, Commands.KEYCODE_COMMIT}
)


def _keypad_code_bytes(frame: bytes | bytearray) -> range:
    """Return the positions of the keypad code in frame, empty when it has none."""
    magic, command = frame[0x00], frame[0x01]
    if magic == 0xBB and command == Commands.GETLOG:
        start = _GETLOG_CODE_START.get(frame[0x04])
        return range(0) if start is None else range(start, 0x10)
    if magic in (0xEE, 0xAA, 0xBB) and command in _KEYCODE_COMMANDS:
        return range(0x04, 0x0B)
    if magic == 0xBB and command == Commands.KEYCODE_GET:
        return range(0x06, 0x0D)
    return range(0)


def _loggable_hex(frame: bytes | bytearray) -> str:
    """Return frame in hex, masking a keypad code and the checksum if MASK_SENSITIVE."""
    if not const.MASK_SENSITIVE or len(frame) != RESPONSE_FRAME_LEN:
        return frame.hex()
    code = _keypad_code_bytes(frame)
    if not code:
        return frame.hex()
    # The checksum at byte 3 gives away the sum of the code's bytes.
    return "".join(
        "**" if index == 0x03 or index in code else f"{byte:02x}"
        for index, byte in enumerate(frame)
    )


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
