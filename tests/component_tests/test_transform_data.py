import pandas as pd
import pytest
from unittest.mock import patch
from src.transform.transform import transform_data

"""
Component Tests for transform_data() Function

These tests validate the transform_data() function as a component,
testing the complete transformation pipeline coordination.

Test Coverage:
1. Full Pipeline Integration: Verifies complete transform pipeline
   (clean → merge) produces expected merged dataset with real data
2. Error Propagation: Tests that failures from cleaning functions are
   properly propagated to the caller
3. Fail-Fast Behaviour: Validates that pipeline stops on first error
   and doesn't execute subsequent steps
4. Component Coordination: Ensures proper sequencing of transformation steps

Component Tests focus on:
- Complete transformation pipeline validation
- Real data processing with merged outcomes
- Error handling and propagation across pipeline stages
- Fail-fast behaviour verification

Note: Function returns filtered high-value customer data with aggregated
total_spent and avg_transaction_value for each customer.
"""


def normalize_nulls(df):
    """Normalize null values to avoid None vs NaN warnings
    in DataFrame comparisons."""
    return df.fillna(pd.NA).replace({pd.NA: None})


@pytest.fixture
def sample_unclean_transactions():
    """Sample unclean transaction data for testing"""
    unclean_transactions = pd.read_csv("data/raw/unclean_transactions.csv")
    return normalize_nulls(unclean_transactions)


@pytest.fixture
def sample_unclean_customers():
    """Sample unclean customer data for testing"""
    unclean_customers = pd.read_csv("data/raw/unclean_customers.csv")
    return normalize_nulls(unclean_customers)


# Removed due to change in requirements (E3 S7) - no filtering
# @pytest.fixture
# def expected_filtered_data_returned():
#     return pd.read_csv("tests/test_data/expected_filtered_clean_results.csv")


@pytest.fixture
def expected_filtered_data_returned():
    """Expected filtered data after transformation"""
    return pd.read_csv("tests/test_data/expected_filtered_clean_results.csv")


def test_transform_data_returns_correct_structure(
    sample_unclean_transactions,
    sample_unclean_customers,
    expected_filtered_data_returned,
):
    """Test complete transformation pipeline with real data"""
    input_data = (sample_unclean_transactions, sample_unclean_customers)
    result = transform_data(input_data)

    # Verify returned DataFrame
    pd.testing.assert_frame_equal(result, expected_filtered_data_returned)


@patch("src.transform.transform.clean_transactions")
def test_transform_data_propagates_transaction_cleaning_exceptions(
    mock_clean_transactions,
):
    """Test that exceptions from clean_transactions are propagated"""
    mock_clean_transactions.side_effect = Exception(
        "Transaction cleaning failed"
    )

    input_data = (pd.DataFrame(), pd.DataFrame())

    with pytest.raises(Exception, match="Transaction cleaning failed"):
        transform_data(input_data)


@patch("src.transform.transform.clean_customers")
@patch("src.transform.transform.clean_transactions")
def test_transform_data_propagates_customer_cleaning_exceptions(
    mock_clean_transactions, mock_clean_customers
):
    """Test that exceptions from clean_customers are propagated"""
    # Mock transactions to succeed, customers to fail
    mock_clean_transactions.return_value = pd.DataFrame(
        {"transaction_id": [1]}
    )
    mock_clean_customers.side_effect = Exception("Customer cleaning failed")

    input_data = (pd.DataFrame(), pd.DataFrame())

    with pytest.raises(Exception, match=r"Customer cleaning failed"):
        transform_data(input_data)


@patch("src.transform.transform.clean_customers")
@patch("src.transform.transform.clean_transactions")
def test_transform_data_handles_transaction_cleaning_failure_first(
    mock_clean_transactions, mock_clean_customers
):
    """Test fail-fast behavior when transaction cleaning fails first"""
    mock_clean_transactions.side_effect = Exception(
        "Transaction cleaning error"
    )

    input_data = (pd.DataFrame(), pd.DataFrame())

    # Should fail on first function call (transactions)
    with pytest.raises(Exception, match="Transaction cleaning error"):
        transform_data(input_data)

    # Customer cleaning should not be called due to fail-fast behaviour
    mock_clean_customers.assert_not_called()
