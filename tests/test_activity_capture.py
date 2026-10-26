import logging
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from yalexs_ble import const
from yalexs_ble.activity_capture import log_record
from yalexs_ble.const import DoorStatus, LockStatus

# The capture's clock: 2026-10-24 12:00:00.250 in Sydney.
_CAPTURED = 1792803600.25

# Captured GetLog replies from a Yale YRD256 whose clock reads 2056: a
# LockOperation (0x00) record and the GetLogError (0x80) that ends the log.
_LOCK_OPERATION = "bb2d0056000b0500a0a176a3df14194a0200"
_GET_LOG_ERROR = "bb2d00de800200009aa176a3a8a176a30200"
# A GetLog reply captured from a Yale Unity on 2026-10-25: a CredentialOperation
# (0x0E) with an RF credential in slot 7 and no credential error.
_CREDENTIAL_OPERATION = "bb2d00300e51addd6a01070013421f170200"
# Built GetLog replies with the time stamp 0x12345678: a log_type neither
# numbering decodes (0x11), a ProductMessage that August's numbering reads as an
# InvalidCredentialEntry (0x17), a CodeAddedv2 (0x0A), and a KeypadOpen (0x01)
# with the code 123456.
_NO_DECODER = "bb2d00e31178563412000000100000000000"
_PRODUCT_MESSAGE = "bb2d00981778563412341201020304050000"
_CODE_ADDED_V2 = "bb2d00d40a78563412010240e20100000000"
_KEYPAD_OPEN = "bb2d006b0178563412123456ffffffff0000"

# The frames' time stamps in Sydney time, with daylight saving time in November
# 2056 and October 2026 and not in September 1979.
_TIME_2056 = "2056-11-26T21:40:{:02d}.000+11:00"
_TIME_2026 = "2026-10-25T18:18:41.000+11:00"
_TIME_1979 = "1979-09-06T08:51:36.000+10:00"


@pytest.fixture(autouse=True)
def _sydney_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the test in Sydney's time zone, so each local time has one spelling."""
    monkeypatch.setenv("TZ", "AEST-10AEDT,M10.1.0,M4.1.0/3")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    ("frame", "mask_sensitive", "raw", "log_type", "lock_time", "lag", "decodes"),
    [
        (
            _LOCK_OPERATION,
            False,
            _LOCK_OPERATION,
            "0x00",
            _TIME_2056.format(32),
            "-949657231.750",
            [
                (
                    "yale",
                    "LockOperation(op_code=ONE_TOUCH,current_lock_state=LOCKED,"
                    f"key_slot=0,time_stamp={_TIME_2056.format(32)},"
                    "battery_level=5343,temperature=25,error=74,"
                    "battery_level_percent=74)",
                )
            ],
        ),
        (
            _CREDENTIAL_OPERATION,
            False,
            _CREDENTIAL_OPERATION,
            "0x0E",
            _TIME_2026,
            "-109120.750",
            [
                (
                    "yale",
                    f"CredentialOperation(time_stamp={_TIME_2026},credential_type=RF,"
                    "slot=7,credential_error=-,current_lock_state=UNLOCKED,"
                    "current_door_state=CLOSED,battery_level=5919,"
                    "battery_level_percent=66)",
                )
            ],
        ),
        (
            _GET_LOG_ERROR,
            False,
            _GET_LOG_ERROR,
            "0x80",
            "-",
            "-",
            [
                (
                    "yale",
                    "GetLogError(error=2,key_slot=0,log_count=0,dirty_bit=0,"
                    f"dirty_bit_set_time={_TIME_2056.format(26)},"
                    f"dirty_bit_cleared_time={_TIME_2056.format(40)})",
                )
            ],
        ),
        (_NO_DECODER, False, _NO_DECODER, "0x11", "-", "-", [("yale", "LogRecord()")]),
        (
            _PRODUCT_MESSAGE,
            False,
            _PRODUCT_MESSAGE,
            "0x17",
            _TIME_1979,
            "1487383704.250",
            [
                (
                    "yale",
                    f"ProductMessage(time_stamp={_TIME_1979},msg_type=4660,"
                    "data=1:2:3:4:5)",
                ),
                (
                    "august",
                    f"InvalidCredentialEntry(time_stamp={_TIME_1979},cred_type=4660,"
                    "battery_level=513)",
                ),
            ],
        ),
        (
            _CODE_ADDED_V2,
            False,
            _CODE_ADDED_V2,
            "0x0A",
            _TIME_1979,
            "1487383704.250",
            [
                (
                    "yale",
                    f"CodeAddedv2(time_stamp={_TIME_1979},slot=513,"
                    "code=000000000001E240)",
                ),
                (
                    "august",
                    f"CodeAddedv2(time_stamp={_TIME_1979},slot=513,code=123456)",
                ),
            ],
        ),
        (
            _CODE_ADDED_V2,
            True,
            "bb2d00**0a785634120102**********0000",
            "0x0A",
            _TIME_1979,
            "1487383704.250",
            [
                (
                    "yale",
                    f"CodeAddedv2(time_stamp={_TIME_1979},slot=513,code=****)",
                )
            ],
        ),
        (
            _KEYPAD_OPEN,
            True,
            "bb2d00**0178563412**************0000",
            "0x01",
            _TIME_1979,
            "1487383704.250",
            [
                (
                    "yale",
                    f"KeypadOpen(time_stamp={_TIME_1979},keycode=****)",
                )
            ],
        ),
    ],
    ids=[
        "LockOperation",
        "CredentialOperation",
        "GetLogError",
        "LogRecord",
        "ProductMessage",
        "CodeAddedv2",
        "CodeAddedv2-masked",
        "KeypadOpen-masked",
    ],
)
def test_the_capture_line_splits_on_spaces_into_its_fields_in_order(
    frame: str,
    mask_sensitive: bool,
    raw: str,
    log_type: str,
    lock_time: str,
    lag: str,
    decodes: list[tuple[str, str]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One debug line per reply; august appears only where the two decodes differ."""
    monkeypatch.setattr(const, "MASK_SENSITIVE", mask_sensitive)
    with (
        caplog.at_level(logging.DEBUG, logger="yalexs_ble.activity_capture"),
        patch(
            "yalexs_ble.activity_capture.time",
            SimpleNamespace(time=lambda: _CAPTURED),
        ),
    ):
        log_record(
            logging.getLogger("yalexs_ble.activity_capture"),
            "Test Lock",
            bytes.fromhex(frame),
            True,
            1.25,
            LockStatus.LOCKED,
            DoorStatus.CLOSED,
        )

    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    prefix, _, fields = record.getMessage().partition(": activity record ")
    assert prefix == "Test Lock"
    assert [tuple(field.split("=", 1)) for field in fields.split(" ")] == [
        ("captured", "2026-10-24T12:00:00.250+11:00"),
        ("log_type", log_type),
        ("lock_time", lock_time),
        ("lag", lag),
        ("since_change", "1.250"),
        ("lock", "LOCKED"),
        ("door", "CLOSED"),
        ("backlog", "True"),
        ("raw", raw),
        *decodes,
    ]
