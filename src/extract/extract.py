import pandas as pd
from src.extract.extract_transactions import extract_transactions
from src.extract.extract_customers import extract_customers
from src.utils.logging_utils import setup_logger

logger = setup_logger("extract_data", "extract_data.log")


def extract_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info("Starting data extraction process")

        transactions = extract_transactions()
        # Transactions Pandas DataFrame obtained above
        customers = extract_customers()
        # Customers Pandas DataFrame obtained above

        logger.info(
            f"Data extraction completed successfully - "
            f"Transactions: {transactions.shape}, Customers: {customers.shape}"
        )

        return (transactions, customers)

    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        raise
