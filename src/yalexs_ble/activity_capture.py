"""Log every activity log record the library reads, with its decodes.

The capture is on for every lock when this module's logger has its own level set
to DEBUG, and for one lock when the child logger named for its Bluetooth address,
in lower case without colons, has its own level set to DEBUG, for example
yalexs_ble.activity_capture.aabbccddeeff for AA:BB:CC:DD:EE:FF. Each lock's
lines go to its child logger.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import UTC, datetime
from enum import Enum
from urllib.parse import quote

from . import const
from .activity_log import LogRecord, parse_log_event_packet
from .const import DoorStatus, LockStatus, LogBrand
from .util import _loggable_hex

_LOGGER = logging.getLogger(__name__)

# The line gives these as fields of its own.
_FRAME_FIELDS = frozenset({"raw", "log_type"})

# The record fields that hold a keypad code, kept apart from their byte positions
# in util._GETLOG_CODE_START because activity_log imports util.
_CODE_FIELDS = frozenset({"keycode", "code"})


def capture_logger(address: str) -> logging.Logger | None:
    """Return address's capture logger, or None when the capture is off for it."""
    logger = _LOGGER.getChild(address.replace(":", "").lower())
    if logging.DEBUG in (_LOGGER.level, logger.level):
        return logger
    return None


def log_capture_on(logger: logging.Logger, name: str) -> None:
    """Log that the capture of name's records has started."""
    logger.debug("%s: capture on", name)


def log_drain_ask(logger: logging.Logger, name: str, delay: float) -> None:
    """Log the capture's ask for a drain after delay seconds."""
    logger.debug("%s: no GetLogError for %d s, asking for a drain", name, delay)


def log_record(
    logger: logging.Logger,
    name: str,
    reply: bytes,
    backlog: bool,
    since_change: float | None,
    lock: LockStatus,
    door: DoorStatus,
) -> None:
    """Log one line for reply with the time logged, lock and door status, and decodes.

    since_change is the seconds since the last settled change of the displayed
    lock status, or None when there has been none.
    """
    captured = time.time()
    yale = _decode(reply, LogBrand.YALE)
    time_stamp: int | None = getattr(yale, "time_stamp", None)
    yale_field = _decoded_field("yale", yale)
    august_field = _decoded_field("august", _decode(reply, LogBrand.AUGUST))
    fields = [
        ("captured", _time(captured)),
        ("log_type", f"0x{reply[0x04]:02X}"),
        ("lock_time", "-" if time_stamp is None else _time(time_stamp)),
        ("lag", "-" if time_stamp is None else f"{captured - time_stamp:.3f}"),
        ("since_change", "-" if since_change is None else f"{since_change:.3f}"),
        ("lock", lock.name),
        ("door", door.name),
        ("backlog", str(backlog)),
        ("raw", _loggable_hex(reply)),
        yale_field,
    ]
    if august_field[1] != yale_field[1]:
        fields.append(august_field)
    logger.debug(
        "%s: activity record %s",
        name,
        " ".join(f"{key}={value}" for key, value in fields),
    )


def _decode(reply: bytes, brand: LogBrand) -> LogRecord:
    """Decode reply in brand's numbering."""
    return parse_log_event_packet(reply, brand)


def _decoded_field(key: str, decoded: LogRecord) -> tuple[str, str]:
    """Return key and the record formatted as one token."""
    values = ",".join(
        f"{field.name}={_value(field.name, getattr(decoded, field.name))}"
        for field in dataclasses.fields(decoded)
        if field.name not in _FRAME_FIELDS
    )
    return key, f"{type(decoded).__name__}({values})"


def _value(name: str, value: object) -> str:
    """Format the value of the record field name as one token."""
    if value is None:
        return "-"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, int) and (name == "time_stamp" or name.endswith("_time")):
        return _time(value)
    if isinstance(value, str):
        if const.MASK_SENSITIVE and name in _CODE_FIELDS:
            return "****"
        return _text(value)
    if isinstance(value, tuple):
        return ":".join(_value(name, item) for item in value)
    return str(value)


def _text(value: str) -> str:
    """Percent-encode value, so no space, comma, equals sign, or parenthesis is left."""
    return quote(value, safe="")


def _time(seconds: float) -> str:
    """Format seconds since 1970 UTC as local ISO 8601 time with milliseconds."""
    return (
        datetime.fromtimestamp(seconds, UTC)
        .astimezone()
        .isoformat(timespec="milliseconds")
    )
