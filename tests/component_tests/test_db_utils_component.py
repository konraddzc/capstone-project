import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from src.utils.db_utils import get_db_connection, DatabaseConnectionError


@pytest.fixture
def test_connection_params():
    return {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_password",
        "host": "testhost",
        "port": "1234",
    }


def test_get_db_connection_full_component_success(test_connection_params):
    """Test successful interaction between create_db_engine and
    get_db_connection components."""
    # Arrange
    mock_connection = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_connection

    with patch("src.utils.db_utils.create_db_engine", return_value=mock_engine) as mock_create_engine, \
         patch("src.utils.db_utils.logger") as mock_logger:
        # Act
        connection = get_db_connection(test_connection_params)

        # Assert
        mock_create_engine.assert_called_once_with(test_connection_params)
        mock_engine.connect.assert_called_once()
        assert connection == mock_connection
        # Verify both functions logged success
        assert mock_logger.info.call_count == 1


def test_get_db_connection_engine_creation_fails(test_connection_params):
    """Test component behaviour when create_db_engine fails with missing
    parameters."""
    # Arrange - missing dbname
    invalid_params = test_connection_params.copy()
    invalid_params["dbname"] = ""

    with patch("src.utils.db_utils.logger") as mock_logger:
        # Act & Assert
        with pytest.raises(DatabaseConnectionError) as excinfo:
            get_db_connection(invalid_params)

        assert "Invalid Connection Parameters: dbname not provided" in str(
            excinfo.value
        )
        mock_logger.error.assert_called_once()


def test_get_db_connection_engine_connects_but_connection_fails(test_connection_params):
    """Test component behaviour when engine is created but connection fails."""
    # Arrange
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = OperationalError(
        "Connection timeout", None, Exception("Original error")
    )

    with patch("src.utils.db_utils.create_db_engine", return_value=mock_engine) as mock_create_engine, \
         patch("src.utils.db_utils.logger") as mock_logger:
        # Act & Assert
        with pytest.raises(DatabaseConnectionError) as excinfo:
            get_db_connection(test_connection_params)

        assert "Operational error when connecting to the database" in str(
            excinfo.value
        )
        mock_create_engine.assert_called_once_with(test_connection_params)
        mock_engine.connect.assert_called_once()
        assert mock_logger.error.call_count == 1


def test_get_db_connection_full_failure_chain(test_connection_params):
    """Test component behaviour with SQLAlchemy error during engine creation."""
    # Arrange
    with patch("src.utils.db_utils.create_db_engine", side_effect=DatabaseConnectionError("Failed to connect to the database: Database driver not found")) as mock_create_engine, \
         patch("src.utils.db_utils.logger") as mock_logger:
        # Act & Assert
        with pytest.raises(DatabaseConnectionError) as excinfo:
            get_db_connection(test_connection_params)

        assert "Failed to connect to the database" in str(excinfo.value)
        mock_create_engine.assert_called_once_with(test_connection_params)


@pytest.mark.parametrize("missing_param", ["dbname", "user", "host", "port"])
def test_get_db_connection_parameter_validation_components(
    test_connection_params, missing_param
):
    """Test parameter validation behaviour across both components."""
    # Arrange
    invalid_params = test_connection_params.copy()
    invalid_params[missing_param] = ""

    with patch("src.utils.db_utils.logger") as mock_logger:
        # Act & Assert
        with pytest.raises(DatabaseConnectionError) as excinfo:
            get_db_connection(invalid_params)

        assert f"{missing_param} not provided" in str(excinfo.value)
        mock_logger.error.assert_called_once()