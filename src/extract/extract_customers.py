import os
import logging
import pandas as pd
import timeit
from src.utils.logging_utils import setup_logger, log_extract_success

# Define the file path for the customers CSV file
FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "raw",
    "unclean_customers.csv",
)

# Configure the logger
logger = setup_logger(__name__, "extract_data.log", level=logging.DEBUG)

EXPECTED_PERFORMANCE = 0.0001

TYPE = "CUSTOMERS from CSV"


def extract_customers() -> pd.DataFrame:
    """
    Extract customer data from CSV file with performance logging.

    Returns:
        DataFrame containing customer records from CSV file.

    Raises:
        Exception: If CSV file cannot be loaded.
    """
    start_time = timeit.default_timer()

    try:
        customers = pd.read_csv(FILE_PATH)
        extract_customers_execution_time = timeit.default_timer() - start_time
        log_extract_success(
            logger,
            TYPE,
            customers.shape,
            extract_customers_execution_time,
            EXPECTED_PERFORMANCE,
        )
        return customers
    except Exception as e:
        logger.error(f"Error loading {FILE_PATH}: {e}")
        raise Exception(f"Failed to load CSV file: {FILE_PATH}")
