import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import ArgumentError, OperationalError, SQLAlchemyError
from src.utils.db_utils import (
    create_db_engine,
    DatabaseConnectionError,
    get_db_connection,
)


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch("src.utils.db_utils.logger")


@pytest.fixture
def test_connection_parameters():
    return {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_password",
        "host": "test_host",
        "port": "1234",
        "schema": "test_schema",
    }


def test_get_db_connection_success(mocker, test_connection_parameters):
    # Arrange
    mock_connection = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_connection

    mock_get_engine = mocker.patch(
        "src.utils.db_utils.create_db_engine", return_value=mock_engine
    )

    # Act
    connection = get_db_connection(test_connection_parameters)

    # Assert
    mock_get_engine.assert_called_once_with(test_connection_parameters)
    mock_engine.connect.assert_called_once()
    assert connection == mock_connection


def test_get_db_connection_failure(mocker, test_connection_parameters):
    mock_connect = mocker.patch(
        "src.utils.db_utils.create_engine",
        side_effect=SQLAlchemyError("Connection error"),
    )
    mock_logger = mocker.patch("src.utils.db_utils.logger")

    expected_connection_string = (
        f"postgresql+psycopg2://{test_connection_parameters['user']}:"
        f"{test_connection_parameters['password']}@"
        f"{test_connection_parameters['host']}:"
        f"{int(test_connection_parameters['port'])}/"
        f"{test_connection_parameters['dbname']}"
    )
    expected_connect_args = {"options": f"-csearch_path={test_connection_parameters['schema']}"}

    with pytest.raises(DatabaseConnectionError) as excinfo:
        get_db_connection(test_connection_parameters)

    assert str(excinfo.value) == (
        "Failed to connect to the database: Connection error"
    )
    mock_connect.assert_called_once_with(expected_connection_string, connect_args=expected_connect_args)
    mock_logger.error.assert_called_once_with(
        "Failed to connect to the database: Connection error"
    )


def test_get_db_connection_timeout_failure(mocker, test_connection_parameters):
    mocker.patch(
        "src.utils.db_utils.create_engine",
        side_effect=OperationalError("timeout", None, Exception("original")),
    )
    mock_logger = mocker.patch("src.utils.db_utils.logger")

    with pytest.raises(DatabaseConnectionError) as excinfo:
        get_db_connection(test_connection_parameters)

    print(str(excinfo.value))
    assert "timeout" in str(excinfo.value)

    mock_logger.error.assert_called_once_with(str(excinfo.value))


def test_create_db_engine_success(
    mocker, mock_logger, test_connection_parameters
):
    # Arrange
    mock_engine = MagicMock()
    mock_create_engine = mocker.patch(
        "src.utils.db_utils.create_engine", return_value=mock_engine
    )
    expected_connection_string = (
        f"postgresql+psycopg2://{test_connection_parameters['user']}:"
        f"{test_connection_parameters['password']}@"
        f"{test_connection_parameters['host']}:"
        f"{int(test_connection_parameters['port'])}/"
        f"{test_connection_parameters['dbname']}"
    )
    expected_connect_args = {"options": f"-csearch_path={test_connection_parameters['schema']}"}

    # Act
    engine = create_db_engine(test_connection_parameters)

    # assert
    mock_create_engine.assert_called_once_with(expected_connection_string, connect_args=expected_connect_args)
    assert engine == mock_engine
    # assert that logger.info was called once with the correct message
    mock_logger.info.assert_called_once_with(
        "Successfully created the database engine."
    )


# Define a set of connection parameters that are invalid
# These should be used as parameterised inputs for the test
# The function should through a ValueError if connection parameters are invalid
@pytest.mark.parametrize(
    "invalid_params",
    [
        {
            "dbname": "",
            "user": "test_user",
            "password": "test_password",
            "host": "test_host",
            "port": "1234",
            "schema": "test_schema",
        },
        {
            "dbname": "test_db",
            "user": "",
            "password": "test_password",
            "host": "test_host",
            "port": "1234",
            "schema": "test_schema",
        },
        {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "host": "",
            "port": "1234",
            "schema": "test_schema",
        },
        {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "host": "test_host",
            "port": "",
            "schema": "test_schema",
        },
        {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "host": "test_host",
            "port": "NaN",
            "schema": "test_schema",
        },
    ],
    ids=[
        "empty_dbname",
        "empty_user",
        "empty_host",
        "empty_port",
        "invalid_port",
    ],
)
def test_create_db_engine_fail_missing_params(mock_logger, invalid_params):
    # Arrange
    # Act
    with pytest.raises(DatabaseConnectionError) as excinfo:
        create_db_engine(invalid_params)

    # Assert
    assert "Invalid Connection Parameters" in str(excinfo.value)
    mock_logger.error.assert_called_once_with(str(excinfo.value))


@pytest.mark.parametrize(
    "exception,message",
    [
        (ArgumentError("Test Argument Error"), "Test Argument Error"),
        (ImportError("Test Import Error"), "Test Import Error"),
    ],
)
def test_create_db_engine_exception_catch(
    mocker, mock_logger, test_connection_parameters, exception, message
):
    mocker.patch("src.utils.db_utils.create_engine", side_effect=exception)

    with pytest.raises(DatabaseConnectionError) as excinfo:
        create_db_engine(test_connection_parameters)

    assert message in str(excinfo.value)
    mock_logger.error.assert_called_once_with(str(excinfo.value))


def test_get_db_connection_generic_exception(mocker, test_connection_parameters):
    """Test that generic exceptions in get_db_connection are properly handled."""
    # Mock create_db_engine to raise a generic exception
    mocker.patch(
        "src.utils.db_utils.create_db_engine", 
        side_effect=RuntimeError("Unexpected error")
    )
    
    with pytest.raises(Exception) as excinfo:
        get_db_connection(test_connection_parameters)
    
    assert "An error occurred: Unexpected error" in str(excinfo.value)
