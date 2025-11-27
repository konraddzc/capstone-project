import os
import pandas as pd
import pytest
from src.transform.clean_customers import clean_customers

"""
Component tests for customer transformation pipeline.

These tests validate that all customer transformation functions work together
correctly to produce the expected cleaned dataset. Tests cover integration
scenarios, error handling, and edge cases that ensure the transformation
pipeline is robust for production use.

Test Categories:
- Integration testing: Validates end-to-end transformation pipeline
- Error handling: Tests behaviour with invalid or malformed input
- Edge cases: Empty data, filtered data, missing schema scenarios
"""


EXPECTED_CLEANED_CUSTOMER_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_data",
    "expected_customers_clean_results.csv",
)

UNCLEAN_CUSTOMER_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "raw",
    "unclean_customers.csv",
)


@pytest.fixture
def expected_cleaned_customers():
    """Fixture to load expected cleaned customer data."""
    return pd.read_csv(EXPECTED_CLEANED_CUSTOMER_DATA_PATH)


@pytest.fixture
def unclean_customers():
    """Fixture to load unclean customer data."""
    return pd.read_csv(UNCLEAN_CUSTOMER_DATA_PATH)


def test_transform_customers_returns_expected_data(
    expected_cleaned_customers, unclean_customers
):
    """
    Test that the transformation function returns the expected cleaned data."""

    # Call the transformation function
    cleaned_customers = clean_customers(unclean_customers)

    # Verify the 2 dataframes are the same of the transformed data
    pd.testing.assert_frame_equal(
        cleaned_customers, expected_cleaned_customers
    )


def test_transform_customers_handles_empty_dataframe():
    """Test that clean_customers handles empty DataFrame appropriately.
    
    Note: This test expects KeyError due to missing required columns.
    """
    empty_df = pd.DataFrame()

    with pytest.raises(KeyError):
        clean_customers(empty_df)


def test_transform_customers_handles_empty_rows_with_columns():
    """Test that clean_customers handles DataFrame with columns but no rows."""
    empty_with_columns = pd.DataFrame(
        columns=["customer_id", "name", "country", "is_active", "age"]
    )

    result = clean_customers(empty_with_columns)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "age" not in result.columns


def test_transform_customers_handles_all_rows_filtered():
    """Test that clean_customers handles DataFrame where
    all rows are filtered out."""
    all_invalid = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "name": ["John", "Jane"],
            "country": [None, None],  # Will be filtered out
            "is_active": ["active", "inactive"],
            "age": [25, 30],
        }
    )

    result = clean_customers(all_invalid)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "age" not in result.columns


def test_transform_customers_missing_required_column():
    """Test that clean_customers handles missing required columns appropriately.
    
    Note: This test expects KeyError when required 'country' column is missing.
    """
    missing_country = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "name": ["John", "Jane"],
            "is_active": ["active", "inactive"],
            "age": [25, 30],
            # Missing 'country' column
        }
    )

    with pytest.raises(KeyError):
        clean_customers(missing_country)
