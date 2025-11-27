import pandas as pd
from unittest.mock import patch
from src.transform.clean_customers import (
    clean_customers,
    remove_missing_values,
    remove_age_column,
    standardise_is_active_column,
)


class TestRemoveMissingValues:
    def test_remove_missing_values_drops_rows_with_missing_country(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "country": ["USA", None, "UK"],
                "is_active": ["1", "0", "active"],
            }
        )
        result = remove_missing_values(df)
        assert len(result) == 2
        assert result["customer_id"].tolist() == [1, 3]

    def test_remove_missing_values_drops_rows_with_missing_is_active(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "country": ["USA", "Canada", "UK"],
                "is_active": ["1", None, "active"],
            }
        )
        result = remove_missing_values(df)
        assert len(result) == 2
        assert result["customer_id"].tolist() == [1, 3]

    def test_remove_missing_values_keeps_complete_rows(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "country": ["USA", "UK"],
                "is_active": ["1", "active"],
            }
        )
        result = remove_missing_values(df)
        assert len(result) == 2


class TestRemoveAgeColumn:
    def test_remove_age_column_drops_age_column(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "name": ["John", "Jane"],
                "age": [25, 30],
                "country": ["USA", "UK"],
            }
        )
        result = remove_age_column(df)
        assert "age" not in result.columns
        assert list(result.columns) == ["customer_id", "name", "country"]

    def test_remove_age_column_preserves_other_columns(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "name": ["John", "Jane"],
                "age": [25, 30],
                "country": ["USA", "UK"],
            }
        )
        result = remove_age_column(df)
        assert len(result) == 2
        assert result["customer_id"].tolist() == [1, 2]


class TestStandardiseIsActiveColumn:
    def test_standardise_is_active_column_converts_values(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4],
                "is_active": ["1", "active", "0", "inactive"],
            }
        )
        result = standardise_is_active_column(df)
        assert result["is_active"].tolist() == [True, True, False, False]

    def test_standardise_is_active_column_preserves_other_columns(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "name": ["John", "Jane"],
                "is_active": ["1", "0"],
            }
        )
        result = standardise_is_active_column(df)
        assert result["customer_id"].tolist() == [1, 2]
        assert result["name"].tolist() == ["John", "Jane"]


class TestCleanCustomers:
    @patch("src.transform.clean_customers.save_dataframe_to_csv")
    def test_clean_customers_full_pipeline(self, mock_save):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 4],
                "name": ["John", "Jane", "Bob", "Alice", "Alice"],
                "age": [25, 30, None, 35, 35],
                "country": ["USA", None, "UK", "Canada", "Canada"],
                "is_active": ["1", "active", "0", "inactive", "inactive"],
            }
        )

        result = clean_customers(df)

        # Should remove row with missing country (Jane)
        # Should remove age column
        # Should standardise is_active values
        # Should remove duplicate row (Alice)
        assert len(result) == 3
        assert "age" not in result.columns
        assert result["is_active"].dtype == bool
        assert mock_save.called

    @patch("src.transform.clean_customers.save_dataframe_to_csv")
    def test_clean_customers_calls_save_function(self, mock_save):
        df = pd.DataFrame(
            {
                "customer_id": [1],
                "name": ["John"],
                "age": [25],
                "country": ["USA"],
                "is_active": ["1"],
            }
        )

        clean_customers(df)

        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[1] == "data/processed"
        assert args[2] == "cleaned_customers.csv"
