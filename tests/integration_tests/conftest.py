import pytest
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from config.db_config import load_db_config

# Load test environment variables
project_root = Path(__file__).parent.parent.parent
test_env_path = project_root / ".env.test"
load_dotenv(test_env_path)


@pytest.fixture(scope="session", autouse=True)
def setup_test_transactions():
    """
    Set up the transactions table before integration tests in this module.
    This fixture runs once per test module for better performance.
    """
    config = load_db_config()
    source_db_config = config["source_database"]

    connection_string = (
        f"postgresql+psycopg2://{source_db_config['user']}"
        f":{source_db_config['password']}@{source_db_config['host']}"
        f":{source_db_config['port']}/{source_db_config['dbname']}"
    )

    engine = create_engine(
        connection_string,
        connect_args={
            "options": f"-csearch_path="
            f"{source_db_config.get('schema', 'public')}"
        },
    )
    sql_file_path = project_root / "data" / "raw" / "unclean_transactions.sql"

    try:
        with open(sql_file_path, "r") as file:
            sql_content = file.read()

        with engine.connect() as connection:
            # Drop table if exists and recreate
            connection.execute(text("DROP TABLE IF EXISTS transactions;"))
            connection.execute(text(sql_content))
            connection.commit()

    except Exception as e:
        # If there's an error, ensure we clean up properly
        with engine.connect() as connection:
            connection.rollback()
        raise RuntimeError(f"Failed to setup test transactions table: {e}")

    # Yield control to tests
    yield


@pytest.fixture(scope="function")
def clean_target_table():
    """
    Clean the target transactions_by_customers table before each test.
    This ensures test isolation by starting with a clean state.
    """
    from src.utils.db_utils import get_db_connection

    config = load_db_config()["target_database"]
    connection = get_db_connection(config)

    try:
        connection.execute(
            text("DROP TABLE IF EXISTS transactions_by_customers")
        )
        connection.commit()
    except Exception:
        # Ignore errors if table doesn't exist
        pass
    finally:
        connection.close()

    yield
