import os
import pandas as pd
import pytest
from src.transform.clean_transactions import clean_transactions

"""Component tests for transaction transformation pipeline.

These tests validate that all transaction transformation functions work together
correctly to produce the expected cleaned dataset. Tests cover integration
scenarios, error handling, and edge cases that ensure the transformation
pipeline is robust for production use.

Test Categories:
- Integration testing: Validates end-to-end transformation pipeline
- Error handling: Tests behaviour with invalid or malformed input
- Edge cases: Empty data, filtered data, missing schema scenarios

Fixtures:
- expected_cleaned_transactions: Expected output from transformation pipeline
- unclean_transactions: Raw input data for transformation testing
"""


EXPECTED_CLEANED_TRANSACTIONS_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_data",
    "expected_transactions_clean_results.csv",
)

UNCLEAN_TRANSACTIONS_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "raw",
    "unclean_transactions.csv",
)


@pytest.fixture
def expected_cleaned_transactions():
    """Fixture to load expected cleaned transactions data."""
    return pd.read_csv(EXPECTED_CLEANED_TRANSACTIONS_DATA_PATH)


@pytest.fixture
def unclean_transactions():
    """Fixture to load unclean transactions data."""
    return pd.read_csv(UNCLEAN_TRANSACTIONS_DATA_PATH)


def test_transform_transactions_returns_expected_data(
    expected_cleaned_transactions, unclean_transactions
):
    """
    Test that clean_transactions integrates:
     - missing value removal,
     - date standardisation,
     - amount conversion,
     - and duplicate removal to match expected cleaned results.
    """

    # Call the transformation function
    cleaned_transactions = clean_transactions(unclean_transactions)

    # Verify the 2 dataframes are the same of the transformed data
    pd.testing.assert_frame_equal(
        cleaned_transactions, expected_cleaned_transactions
    )


def test_transform_transactions_handles_empty_dataframe():
    """Test that clean_transactions handles empty DataFrame appropriately.
    
    Note: This test expects KeyError due to missing required columns.
    """
    empty_df = pd.DataFrame()

    with pytest.raises(KeyError):
        clean_transactions(empty_df)


def test_transform_transactions_handles_empty_rows_with_columns():
    """Test that clean_transactions handles DataFrame with columns but no rows.
    
    Note: This test expects AttributeError due to datetime processing on empty data.
    """
    empty_with_columns = pd.DataFrame(
        columns=["transaction_id", "customer_id", "transaction_date", "amount"]
    )

    with pytest.raises(AttributeError):
        clean_transactions(empty_with_columns)


def test_transform_transactions_handles_all_rows_filtered():
    """Test that clean_transactions handles all invalid data appropriately.
    
    Note: This test expects AttributeError when processing null datetime values.
    """
    all_invalid = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "customer_id": [101, 102],
            "transaction_date": [None, None],  # Will cause AttributeError
            "amount": [100.0, 200.0],
        }
    )

    with pytest.raises(AttributeError):
        clean_transactions(all_invalid)


def test_transform_transactions_missing_required_column():
    """Test that clean_transactions handles missing required columns appropriately.
    
    Note: This test expects KeyError when required 'transaction_date' column is missing.
    """
    missing_date = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "customer_id": [101, 102],
            "amount": [100.0, 200.0],
            # Missing 'transaction_date' column
        }
    )

    with pytest.raises(KeyError):
        clean_transactions(missing_date)
