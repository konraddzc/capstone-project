import pandas as pd
import pytest
from unittest.mock import patch
from src.transform.merge_customers_transactions import (
    merge_transactions_customers,
)


@pytest.fixture
def mock_save():
    """Fixture to provide mocked save function."""
    with patch(
        "src.transform.merge_customers_transactions.save_dataframe_to_csv"
    ) as mock:
        yield mock


@pytest.fixture
def sample_transactions():
    """Fixture providing sample transaction data."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "customer_id": [101, 102, 103],
            "transaction_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [100.0, 200.0, 300.0],
        }
    )


@pytest.fixture
def sample_customers():
    """Fixture providing sample customer data."""
    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "name": ["Alice", "Bob", "Charlie"],
            "country": ["USA", "UK", "Canada"],
            "is_active": [True, False, True],
        }
    )


class TestMergeTransactionsCustomers:
    """Unit tests for the merge_transactions_customers function."""

    def test_successful_merge(
        self, mock_save, sample_transactions, sample_customers
    ):
        """Test successful merge of transactions and customers."""
        result = merge_transactions_customers(
            sample_transactions, sample_customers
        )

        assert len(result) == 3
        assert set(result.columns) == {
            "transaction_id",
            "customer_id",
            "transaction_date",
            "amount",
            "name",
            "country",
            "is_active",
        }
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
        mock_save.assert_called_once()

    def test_partial_match(self, mock_save):
        """Test merge with partial customer matches."""
        transactions = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3],
                "customer_id": [101, 102, 999],  # 999 has no customer match
                "transaction_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "amount": [100.0, 200.0, 300.0],
            }
        )

        customers = pd.DataFrame(
            {
                "customer_id": [101, 102],
                "name": ["Alice", "Bob"],
                "country": ["USA", "UK"],
                "is_active": [True, False],
            }
        )

        result = merge_transactions_customers(transactions, customers)

        assert len(result) == 2
        assert result["customer_id"].tolist() == [101, 102]

    def test_empty_dataframes(self, mock_save):
        """Test merge with empty DataFrames."""
        empty_transactions = pd.DataFrame(
            columns=[
                "transaction_id",
                "customer_id",
                "transaction_date",
                "amount",
            ]
        )
        empty_customers = pd.DataFrame(
            columns=["customer_id", "name", "country", "is_active"]
        )

        result = merge_transactions_customers(
            empty_transactions, empty_customers
        )

        assert len(result) == 0
        assert len(result.columns) == 7

    def test_duplicate_transactions(self, mock_save):
        """Test merge with duplicate customer_ids in transactions."""
        transactions = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3],
                "customer_id": [101, 101, 102],
                "transaction_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "amount": [100.0, 150.0, 200.0],
            }
        )

        customers = pd.DataFrame(
            {
                "customer_id": [101, 102],
                "name": ["Alice", "Bob"],
                "country": ["USA", "UK"],
                "is_active": [True, False],
            }
        )

        result = merge_transactions_customers(transactions, customers)

        assert len(result) == 3
        assert result["name"].tolist() == ["Alice", "Alice", "Bob"]
        assert result["customer_id"].tolist() == [101, 101, 102]
