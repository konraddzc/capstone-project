import pytest
import pandas as pd
from src.transform.filter_customer_transactions import (
    filter_for_high_value_customers,
    find_high_value_total_spend,
    find_avg_transaction_amount,
    get_high_value_customers,
)


@pytest.fixture
def merged_data():
    """Load test data from CSV file."""
    return pd.read_csv("tests/test_data/expected_merged_clean_results.csv")


@pytest.fixture
def expected_filtered_total_spent():
    """Load expected total spend results from CSV file."""
    return pd.read_csv("tests/test_data/expected_filtered_total_spent.csv")


@pytest.fixture
def expected_avg_transaction_value():
    """Load expected average transaction value results from CSV file."""
    return pd.read_csv("tests/test_data/expected_avg_transaction_value.csv")


@pytest.fixture
def expected_filtered_data():
    """Load expected filtered results from CSV file."""
    return pd.read_csv("tests/test_data/expected_filtered_clean_results.csv")


def test_filter_for_high_value_customers(merged_data, expected_filtered_data):
    """Test the main filter function returns expected results."""
    result = filter_for_high_value_customers(merged_data)

    pd.testing.assert_frame_equal(
        result.sort_values("customer_id").reset_index(drop=True),
        expected_filtered_data.sort_values("customer_id").reset_index(
            drop=True
        ),
        check_like=True,
    )


def test_find_high_value_total_spend(
    merged_data, expected_filtered_total_spent
):
    """Test finding customers with total spend above threshold."""
    result = find_high_value_total_spend(merged_data, 500)

    pd.testing.assert_frame_equal(
        result.sort_values("customer_id").reset_index(drop=True),
        expected_filtered_total_spent.sort_values("customer_id").reset_index(
            drop=True
        ),
        check_like=True,
    )


def test_find_avg_transaction_amount(
    merged_data, expected_avg_transaction_value
):
    """Test calculating average transaction amount per customer."""
    result = find_avg_transaction_amount(merged_data)

    pd.testing.assert_frame_equal(
        result.sort_values("customer_id").reset_index(drop=True),
        expected_avg_transaction_value.sort_values("customer_id").reset_index(
            drop=True
        ),
        check_like=True,
    )


def test_get_high_value_customers():
    """Test merging total spend and average transaction DataFrames."""
    total_spend = pd.DataFrame(
        {"customer_id": [1, 2, 3], "amount": [600, 700, 800]}
    )
    avg_transaction = pd.DataFrame(
        {"customer_id": [1, 2, 3], "avg_transaction_amount": [200, 350, 400]}
    )

    result = get_high_value_customers(total_spend, avg_transaction)

    # Create expected result
    expected = pd.merge(total_spend, avg_transaction, on="customer_id")

    pd.testing.assert_frame_equal(result, expected)
