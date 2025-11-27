import logging
import os
import pandas as pd
import timeit
from config.db_config import load_db_config
from src.extract.extract_query import execute_extract_query
from src.utils.sql_utils import import_sql_query
from src.utils.db_utils import get_db_connection
from src.utils.logging_utils import setup_logger, log_extract_success

# Configure the logger
logger = setup_logger(__name__, "extract_data.log", level=logging.DEBUG)

EXTRACT_TRANSACTIONS_QUERY_FILE = os.path.join(
    os.path.dirname(__file__), "../sql/extract_transactions.sql"
)

EXPECTED_IMPORT_RATE = 0.001

TYPE = "TRANSACTIONS from database"


def extract_transactions() -> pd.DataFrame:
    """
    Extract transactions data with performance logging.

    Returns:
        DataFrame containing transaction records.
    """
    try:
        # Set up performance recording for transaction extraction
        start_time = timeit.default_timer()
        transactions = extract_transactions_execution()
        extract_transactions_execution_time = (
            timeit.default_timer() - start_time
        )

        log_extract_success(
            logger,
            TYPE,
            transactions.shape,
            extract_transactions_execution_time,
            EXPECTED_IMPORT_RATE,
        )
        return transactions
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        raise Exception(f"Failed to extract data: {e}")


def extract_transactions_execution() -> pd.DataFrame:
    """
    Execute the transaction extraction query against the database.

    Returns:
        DataFrame containing transaction records from the database.
    """
    # Connect to the database
    # Import the SQL query
    # Execute the query
    # Return the dataframe as a result
    connection_details = load_db_config()["source_database"]
    print(connection_details)
    connection = get_db_connection(connection_details)
    query = import_sql_query(EXTRACT_TRANSACTIONS_QUERY_FILE)
    transactions_df = execute_extract_query(query, connection)
    connection.close()
    # print(transactions_df)
    # Initially added to debug during dev - remove before production
    return transactions_df
