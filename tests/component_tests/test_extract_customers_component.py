import pandas as pd
import pytest
import timeit
from unittest.mock import patch
from src.extract.extract_customers import (
    extract_customers,
    EXPECTED_PERFORMANCE,
)

"""
Component Tests for extract_customers() Function

These tests validate the extract_customers() function as a complete component,
testing its behaviour with real file system interactions and actual data.

Test Coverage:
1. Data Integrity: Verifies extracted data matches expected raw CSV exactly
2. Performance: Ensures extraction meets <1 second requirement for 5200 rows
3. Error Handling: Tests proper exception handling for missing and corrupt files

Component tests differ from unit tests by:
- Using real file paths and actual CSV data
- Testing the complete function behaviour end-to-end
- Validating integration with file system and pandas library
- Ensuring production-like performance requirements are met
"""


@pytest.fixture
def expected_unclean_customers():
    return pd.read_csv("data/raw/unclean_customers.csv")


def test_extract_customers_returns_correct_dataframe(
    expected_unclean_customers,
):
    # Call the function to get the DataFrame
    df = extract_customers()

    # Verify the DataFrame is the same as the expected unclean customers
    pd.testing.assert_frame_equal(df, expected_unclean_customers)


def test_extract_customers_performance():
    execution_time = timeit.timeit(
        "extract_customers()", globals=globals(), number=1
    )

    # Call the function to get the DataFrame
    df = extract_customers()

    # Load time per row
    actual_execution_time_per_row = execution_time / df.shape[0]

    # Assert that the execution time is within an acceptable range
    assert actual_execution_time_per_row <= EXPECTED_PERFORMANCE, (
        f"Expected execution time to be less than or equal to "
        f"{str(EXPECTED_PERFORMANCE)} seconds, but got "
        f"{str(actual_execution_time_per_row)} seconds"
    )


@patch("src.extract.extract_customers.FILE_PATH", "nonexistent_file.csv")
def test_extract_customers_file_not_found():
    with pytest.raises(Exception, match="Failed to load CSV file"):
        extract_customers()


def test_extract_customers_corrupt_csv():
    corrupt_file_path = "tests/test_data/corrupt_customers.csv"

    with patch("src.extract.extract_customers.FILE_PATH", corrupt_file_path):
        with pytest.raises(Exception, match="Failed to load CSV file"):
            extract_customers()
