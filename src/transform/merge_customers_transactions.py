import pandas as pd
from src.utils.file_utils import save_dataframe_to_csv


OUTPUT_DIR = "data/processed/"
FILE_NAME = "merged_data.csv"


def merge_transactions_customers(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    merged_data = pd.merge(transactions, customers, on="customer_id")

    # Save the merged data to a CSV file
    save_dataframe_to_csv(merged_data, OUTPUT_DIR, FILE_NAME)

    return merged_data
