"""Frames and helpers shared by the lock and push tests."""

from typing import Any

from yalexs_ble.const import RESPONSE_FRAME_LEN
from yalexs_ble.util import _simple_checksum


def bb_frame(opcode: int, byte_4: int, value: int) -> bytes:
    """Build a 0xBB frame with the opcode, byte 4, and the value at bytes 8 to 11."""
    frame = bytearray(RESPONSE_FRAME_LEN)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x04] = byte_4
    frame[0x08:0x0C] = value.to_bytes(4, "little")
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


def call_method(
    target: object, method: str, args: tuple[Any, ...] | dict[str, Any]
) -> Any:
    """Call a method with a tuple as positional or a dict as keyword arguments."""
    if isinstance(args, dict):
        return getattr(target, method)(**args)
    return getattr(target, method)(*args)


# A SetRTC acknowledgment captured on 2026-11-04 from a Yale lock, answering 1793750707.
CLOCK_WRITE_ACK = bytes.fromhex("aa1000c7b376ea6a00000000000000000200")
# Constructed TimeZone_Info frames. Every payload byte is distinct, so a payload
# written at the wrong offset fails, and the reply holds 0x01 at byte 7 where the
# payload and the acknowledgment hold 0xF7, so a return taken from the acknowledgment
# fails. Byte 4 of each is the payload's first byte.
ZONE_PAYLOAD = bytes(range(0xF4, 0x100))
ZONE_ACK = bytes.fromhex("aa300072f4f5f6f7f8f9fafbfcfdfeff0200")
ZONE_REPLY = bytes.fromhex("bb300059f4f5f601f8f9fafbfcfdfeff0000")
