import pytest

from yalexs_ble.util import _bytes_to_signed_int


@pytest.mark.parametrize(
    ("buffer", "value"),
    [
        (bytes.fromhex("f2000000"), 242),
        (bytes.fromhex("ffffff7f"), 0x7FFFFFFF),
        (bytes.fromhex("00000080"), -0x80000000),
        (bytes.fromhex("efbeadde"), -559038737),
    ],
)
def test_bytes_to_signed_int_reads_little_endian_twos_complement(
    buffer: bytes, value: int
) -> None:
    assert _bytes_to_signed_int(buffer) == value
