from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import ArgumentError, OperationalError, SQLAlchemyError
import logging
from typing import Dict, Literal
from src.utils.logging_utils import setup_logger


# Create a Custom Exception for Connection Errors
class DatabaseConnectionError(Exception):
    pass


# Create a Custom Exception for Query Errors
class QueryExecutionError(Exception):
    pass


# Configure the logger
logger = setup_logger(__name__, "database.log", level=logging.DEBUG)


# Need to create database engine
# Tells python what flavour of database it is connecting to
# AND the credentials it needs to create a connection
def create_db_engine(
    connection_params: Dict[
        Literal["dbname", "user", "password", "host", "port", "schema"], str
    ],
) -> Engine:
    """
    Create a SQLAlchemy database engine for PostgreSQL connection.

    Args:
        connection_params: Must include 'dbname', 'user', 'password',
                          'host', 'port'.

    Raises:
        DatabaseConnectionError: If connection parameters are invalid or
                               missing.
    """
    try:
        if not connection_params.get("dbname"):
            raise ValueError("dbname not provided")
        if not connection_params.get("user"):
            raise ValueError("user not provided")
        if not connection_params.get("host"):
            raise ValueError("host not provided")
        if not connection_params.get("port"):
            raise ValueError("port not provided")
        engine = create_engine(
            f"postgresql+psycopg2://{connection_params['user']}"
            f":{connection_params['password']}@"
            f"{connection_params['host']}:"
            f"{connection_params['port']}/"
            f"{connection_params['dbname']}",
            connect_args={
                "options": f"-csearch_path="
                f"{connection_params.get('schema', 'public')}"
            },
        )
        logger.info("Successfully created the database engine.")
        return engine
    except ArgumentError as e:
        logger.error(f"Invalid Connection Parameters: {e}")
        raise DatabaseConnectionError(f"Invalid Connection Parameters: {e}")
    except ImportError as e:
        logger.error(f"Invalid DB Driver: {e}")
        raise DatabaseConnectionError(f"Invalid DB Driver: {e}")
    except ValueError as e:
        logger.error(f"Invalid Connection Parameters: {e}")
        raise DatabaseConnectionError(f"Invalid Connection Parameters: {e}")


# Once we have the engine
# We can create a connection
def get_db_connection(
    connection_params: Dict[
        Literal["dbname", "user", "password", "host", "port", "schema"], str
    ],
) -> Connection:
    """
    Create and return a database connection.

    Raises:
        DatabaseConnectionError: If connection fails.
    """
    try:
        engine = create_db_engine(connection_params)
        connection = engine.connect()
        logger.info("Successfully connected to the database.")
        return connection
    except OperationalError as e:
        logger.error(f"Operational error when connecting to the database: {e}")
        raise DatabaseConnectionError(
            f"Operational error when connecting to the database: {e}"
        )
    except DatabaseConnectionError:
        # Re-raise DatabaseConnectionError from create_db_engine
        raise
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise DatabaseConnectionError(
            f"Failed to connect to the database: {e}"
        )
    except Exception as e:
        # This catches any unhandled exceptions in get_db_connection
        # and create_db_engine
        raise Exception(f"An error occurred: {e}")
