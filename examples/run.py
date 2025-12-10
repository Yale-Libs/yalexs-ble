import asyncio
import logging

from bleak import BleakScanner

from yalexs_ble import LockState, PushLock, serial_to_local_name
from yalexs_ble.const import ConnectionInfo, LockInfo

_LOGGER = logging.getLogger(__name__)

LOCK_SERIAL = None
LOCK_KEY = None
LOCK_KEY_INDEX = None


assert isinstance(LOCK_SERIAL, str)  # nosec
assert isinstance(LOCK_KEY, str)  # type: ignore[unreachable] # nosec
assert isinstance(LOCK_KEY_INDEX, int)  # nosec


async def run():
    push_lock = PushLock(
        serial_to_local_name(LOCK_SERIAL), key=LOCK_KEY, key_index=LOCK_KEY_INDEX
    )
    _LOGGER.info("Expected local_name: %s", push_lock.local_name)

    def new_state(
        new_state: LockState, lock_info: LockInfo, connection_info: ConnectionInfo
    ) -> None:
        _LOGGER.info(
            "New state: %s, lock_info: %s, connection_info: %s",
            new_state,
            lock_info,
            connection_info,
        )

    cancel_callback = push_lock.register_callback(new_state)
    scanner = BleakScanner(detection_callback=push_lock.update_advertisement)
    await scanner.start()
    cancel = await push_lock.start()
    _LOGGER.info(
        "Started, waiting for lock to be discovered with local_name: %s",
        push_lock.local_name,
    )
    await push_lock.wait_for_first_update(10000)
    await asyncio.sleep(1000000)
    cancel_callback()
    cancel()
    await scanner.stop()


logging.basicConfig(level=logging.INFO)
logging.getLogger("yalexs_ble").setLevel(logging.DEBUG)
logging.getLogger("bleak_retry_connector").setLevel(logging.DEBUG)
asyncio.run(run())
