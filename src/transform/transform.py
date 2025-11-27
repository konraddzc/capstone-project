import pandas as pd
from src.transform.clean_transactions import clean_transactions
from src.transform.clean_customers import clean_customers
from src.transform.filter_customer_transactions import (
    filter_for_high_value_customers,
)
from src.transform.merge_customers_transactions import (
    merge_transactions_customers,
)

"""
    Due to the new requirements, this code is longer needed (Epic 3 Story 7)

from src.transform.filter_customer_transactions import (
    filter_for_high_value_customers,
)
"""

from src.utils.logging_utils import setup_logger

logger = setup_logger("transform_data", "transform_data.log")


def transform_data(data) -> pd.DataFrame:
    try:
        logger.info("Starting data transformation process...")
        # Clean transaction data
        logger.info("Cleaning transaction data...")
        cleaned_transactions = clean_transactions(data[0])
        logger.info("Transaction data cleaned successfully.")
        # Clean customer data
        logger.info("Cleaning customer data...")
        cleaned_customers = clean_customers(data[1])
        logger.info("Customer data cleaned successfully.")
        # Enrich and aggregate customer/transaction data
        logger.info("Merging customer and transaction data...")
        merged_data = merge_transactions_customers(
            cleaned_transactions, cleaned_customers
        )
        logger.info("Data merged successfully.")
        # Filter the merged data for the high value customers
        logger.info("Filtering merged data for high value customers...")
        high_value_customers = filter_for_high_value_customers(merged_data)
        logger.info("High value customers filtered successfully.")

        return high_value_customers
    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")
        raise
