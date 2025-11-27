import pytest
from src.extract.extract_transactions import (
    extract_transactions,
    TYPE,
    # EXTRACT_TRANSACTIONS_QUERY_FILE,
    EXPECTED_IMPORT_RATE,
)


@pytest.fixture
def mock_log_extract_success(mocker):
    return mocker.patch("src.extract.extract_transactions.log_extract_success")


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch("src.extract.extract_transactions.logger")


def test_log_extract_success_transactions(
    mocker, mock_log_extract_success, mock_logger
):
    mock_execution_time = 0.5

    # Mock the extract_transactions_execution function to return a DataFrame
    import pandas as pd

    mock_df = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    mocker.patch(
        "src.extract.extract_transactions.extract_transactions_execution",
        return_value=mock_df,
    )

    # Mock timeit.default_timer to control the execution time
    mock_start_time = 100.0
    mock_end_time = 100.5
    mocker.patch(
        "src.extract.extract_transactions.timeit.default_timer",
        side_effect=[mock_start_time, mock_end_time],
    )

    df = extract_transactions()

    # Assertions
    mock_log_extract_success.assert_called_once_with(
        mock_logger,
        TYPE,
        df.shape,
        mock_execution_time,
        EXPECTED_IMPORT_RATE,
    )


def test_log_transactions_error(mocker, mock_logger):
    # Patch the function to raise an exception
    mocker.patch(
        "src.extract.extract_transactions.extract_transactions_execution",
        side_effect=Exception("Exception message"),
    )

    # Test that the exception is raised
    with pytest.raises(Exception, match="Exception message"):
        extract_transactions()

    # Verify that the error was logged
    mock_logger.error.assert_called_once_with(
        "Failed to extract data: Exception message"
    )


def test_extract_transactions_execution_mocked(mocker):
    """Test extract_transactions_execution with all dependencies mocked"""
    import pandas as pd
    from unittest.mock import MagicMock

    # Mock all the dependencies
    mock_db_config = {
        "source_database": {
            "host": "test_host",
            "port": "5432",
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
        }
    }
    mocker.patch(
        "src.extract.extract_transactions.load_db_config",
        return_value=mock_db_config,
    )

    mock_query = "SELECT * FROM transactions"
    mocker.patch(
        "src.extract.extract_transactions.import_sql_query",
        return_value=mock_query,
    )

    mock_connection = MagicMock()
    mocker.patch(
        "src.extract.extract_transactions.get_db_connection",
        return_value=mock_connection,
    )

    mock_df = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    mocker.patch(
        "src.extract.extract_transactions.execute_extract_query",
        return_value=mock_df,
    )

    # Import the function to test
    from src.extract.extract_transactions import extract_transactions_execution

    # Execute the function
    result = extract_transactions_execution()

    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ["id", "amount"]

    # Verify that connection.close() was called
    mock_connection.close.assert_called_once()
