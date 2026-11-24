"""Frames and helpers shared by the lock and push tests."""

import math
from typing import Any

import pytest

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


def with_checksum(hex_str: str) -> bytes:
    """Build a frame from hex with a valid simple checksum in byte 3."""
    frame = bytearray.fromhex(hex_str)
    frame[0x03] = 0
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


def call_method(
    target: object, method: str, args: tuple[Any, ...] | dict[str, Any]
) -> Any:
    """Call a method with a tuple as positional or a dict as keyword arguments."""
    if isinstance(args, dict):
        return getattr(target, method)(**args)
    return getattr(target, method)(*args)


# A SetRTC acknowledgment captured on 2026-11-04 from a Yale lock.
CLOCK_WRITE_ACK = bytes.fromhex("aa1000c7b376ea6a00000000000000000200")
# The time CLOCK_WRITE_ACK answers.
CLOCK_WRITE_TIME = 1793750707
# Constructed TimeZone_Info frames. Every payload byte is distinct, so a payload
# written at the wrong offset fails, and the reply holds 0x01 at byte 7 where the
# payload and the acknowledgment hold 0xF7, so a return taken from the acknowledgment
# fails.
ZONE_PAYLOAD = bytes(range(0xF4, 0x100))
ZONE_ACK = with_checksum(f"aa300000{ZONE_PAYLOAD.hex()}0200")
ZONE_REPLY = with_checksum(
    f"bb300000{ZONE_PAYLOAD[:3].hex()}01{ZONE_PAYLOAD[4:].hex()}0000"
)

# sync_rtc offsets that are not a finite int or float, refused by Lock and PushLock.
CLOCK_OFFSET_BAD_ARGUMENTS = [
    pytest.param(
        "sync_rtc",
        {"offset": "0.5"},
        TypeError,
        "Clock offset has type str, not int or float: '0.5'",
        id="an-offset-that-is-a-str",
    ),
    pytest.param(
        "sync_rtc",
        {"offset": None},
        TypeError,
        "Clock offset has type NoneType, not int or float: None",
        id="an-offset-that-is-none",
    ),
    pytest.param(
        "sync_rtc",
        {"offset": True},
        TypeError,
        "Clock offset has type bool, not int or float: True",
        id="an-offset-that-is-a-bool",
    ),
    pytest.param(
        "sync_rtc",
        {"offset": math.inf},
        ValueError,
        "Clock offset is not finite: inf",
        id="an-infinite-offset",
    ),
    pytest.param(
        "sync_rtc",
        {"offset": -math.inf},
        ValueError,
        "Clock offset is not finite: -inf",
        id="a-negative-infinite-offset",
    ),
    pytest.param(
        "sync_rtc",
        {"offset": math.nan},
        ValueError,
        "Clock offset is not finite: nan",
        id="a-nan-offset",
    ),
]
# A set_timezone_info payload that is not bytes, refused by Lock and PushLock.
ZONE_PAYLOAD_BAD_ARGUMENTS = [
    pytest.param(
        "set_timezone_info",
        ("abcdefghijkl",),
        TypeError,
        "Time zone payload has type str, not bytes or bytearray: 'abcdefghijkl'",
        id="a-payload-that-is-a-str",
    ),
]
