import pytest
from src.transform import transform


@pytest.mark.parametrize(
    "value, expected",
    [
        (123, ""),
        (None, "none"),
    ],
)
def test_process_text_non_string(value, expected):
    result = transform.process_text(value)
    assert isinstance(result, str)
    assert result == expected
