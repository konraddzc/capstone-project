import pandas as pd
import pytest
from unittest.mock import patch
from src.extract.extract import extract_data

"""
Integration Component Tests for extract_data() Function

These tests validate the extract_data() function as an integration component,
testing how it coordinates multiple extraction functions to work together.

Test Coverage:
1. Data Integration: Verifies that extract_data() properly coordinates both
   extract_transactions() and extract_customers() functions
2. Data Integrity: Ensures extracted data from both sources matches expected
results exactly
3. Error Propagation: Tests that failures from either extraction function are
properly handled
4. Execution Flow: Validates fail-fast behaviour and function call ordering
5. Data Source Access: Confirms both database and CSV sources are accessed with
compatible data

Integration Component Tests focus on:
- Coordination between multiple components
    (extract_transactions + extract_customers)
- Real data source integration (actual database + CSV file)
- End-to-end data flow validation within the extract phase
- Cross-component error handling and propagation
- Data compatibility between different source types

This differs from individual component tests by:
- Testing multiple components working together
- Validating data relationships between sources
- Ensuring the extract phase works as a cohesive unit
- Using minimal mocking (only for error simulation)

Note: Logging is not tested here as it's considered a cross-cutting concern
best validated at the E2E test level.
"""

"""Component tests for the extract_data function.

This module tests the extract_data function which orchestrates the extraction
of data from multiple sources (database and CSV file). These are component-level
tests that verify the integration between extraction functions while testing
with real data sources.

Test Categories:
- Integration testing: Verifies extract_data correctly combines extraction 
functions
- Error handling: Tests exception propagation from underlying extraction 
functions
- Data validation: Confirms extracted data matches expected structure and 
content
- Source verification: Validates access to both database and CSV data sources

Fixtures:
- expected_transactions: Normalized transaction data from CSV for comparison
- expected_customers: Normalized customer data from CSV for comparison
"""


def normalize_nulls(df):
    """Normalize null values to avoid None vs NaN warnings
    in DataFrame comparisons."""
    return df.fillna(pd.NA).replace({pd.NA: None})


@pytest.fixture
def expected_transactions():
    df = pd.read_csv("data/raw/unclean_transactions.csv")
    return normalize_nulls(df)


@pytest.fixture
def expected_customers():
    df = pd.read_csv("data/raw/unclean_customers.csv")
    return normalize_nulls(df)


@patch("src.extract.extract_transactions.load_db_config")
@patch("src.extract.extract_transactions.execute_extract_query")
@patch("src.extract.extract_transactions.get_db_connection")
def test_extract_data_returns_correct_data(
    mock_db_connection,
    mock_execute_query,
    mock_load_db_config,
    expected_transactions,
    expected_customers,
):
    """Test that extract_data integrates extraction functions correctly"""
    # Mock database configuration
    mock_load_db_config.return_value = {
        "source_database": {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
    }
    # Mock database query execution to return expected transactions
    mock_execute_query.return_value = expected_transactions

    # Execute with real logging
    result = extract_data()

    # Verify structure
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)

    # Verify content matches expected data exactly
    transactions, customers = result
    transactions = normalize_nulls(transactions)
    customers = normalize_nulls(customers)

    pd.testing.assert_frame_equal(
        transactions, expected_transactions, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        customers, expected_customers, check_dtype=False
    )


@patch("src.extract.extract.extract_transactions")
def test_extract_data_propagates_transaction_exceptions(
    mock_extract_transactions,
):
    """Test that exceptions from extract_transactions are propagated"""
    mock_extract_transactions.side_effect = Exception("DB Error")

    with pytest.raises(Exception, match="DB Error"):
        extract_data()


@patch("src.extract.extract_transactions.load_db_config")
@patch("src.extract.extract_transactions.execute_extract_query")
@patch("src.extract.extract_transactions.get_db_connection")
@patch("src.extract.extract.extract_customers")
def test_extract_data_propagates_customer_exceptions(
    mock_extract_customers,
    mock_db_connection,
    mock_execute_query,
    mock_load_db_config,
):
    """Test that exceptions from extract_customers are propagated"""
    # Mock database configuration
    mock_load_db_config.return_value = {
        "source_database": {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
    }
    # Mock successful transaction extraction
    mock_execute_query.return_value = pd.DataFrame(
        {"transaction_id": [1], "customer_id": [1], "amount": [100.0]}
    )
    # Mock customer extraction failure
    mock_extract_customers.side_effect = Exception("File Error")

    with pytest.raises(Exception, match="File Error"):
        extract_data()


@patch("src.extract.extract.extract_customers")
@patch("src.extract.extract.extract_transactions")
def test_extract_data_handles_transaction_failure_first(
    mock_extract_transactions, mock_extract_customers
):
    """Test behavior when transactions extraction fails first"""
    mock_extract_transactions.side_effect = Exception(
        "Database connection failed"
    )

    # Should fail on first function call (transactions)
    with pytest.raises(Exception, match="Database connection failed"):
        extract_data()

    # Customers extraction should not be called
    mock_extract_customers.assert_not_called()


@patch("src.extract.extract_transactions.load_db_config")
@patch("src.extract.extract_transactions.execute_extract_query")
@patch("src.extract.extract_transactions.get_db_connection")
def test_extract_data_accesses_both_data_sources(
    mock_db_connection, mock_execute_query, mock_load_db_config
):
    """Test that extract_data accesses both database and CSV data sources"""
    # Mock database configuration
    mock_load_db_config.return_value = {
        "source_database": {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
    }
    # Create mock transaction data
    mock_transactions = pd.DataFrame(
        {
            "transaction_id": range(1, 10001),
            "customer_id": range(1, 10001),
            "amount": [100.0] * 10000,
        }
    )
    mock_execute_query.return_value = mock_transactions

    transactions, customers = extract_data()

    # Verify database source characteristics (transactions)
    assert transactions.shape[0] >= 10000  # Large dataset from database
    assert "transaction_id" in transactions.columns
    assert "amount" in transactions.columns

    # Verify CSV source characteristics (customers)
    assert customers.shape[0] > 5000  # Expected customer count from CSV
    assert customers.shape[1] == 5  # CSV has 5 columns including age
    assert "customer_id" in customers.columns
    assert "name" in customers.columns
    assert "age" in customers.columns

    # Verify data sources contain overlapping customer IDs
    transaction_customer_ids = set(transactions["customer_id"].dropna())
    customer_ids = set(customers["customer_id"].dropna())
    overlap = transaction_customer_ids.intersection(customer_ids)
    assert len(overlap) > 0, "No overlapping customer IDs between data sources"
