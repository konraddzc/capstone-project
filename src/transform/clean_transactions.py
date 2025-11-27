import pandas as pd

from src.utils.date_utils import standardise_date
from src.utils.file_utils import save_dataframe_to_csv


OUTPUT_DIR = "data/processed"
FILE_NAME = "cleaned_transactions.csv"


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    # Task 2 - Remove rows with missing values
    transactions = remove_missing_values(transactions)
    # Task 3 - Standardise the date formats
    transactions = standardise_date_format(transactions)
    # Task 4 - Convert amount to numeric
    transactions = convert_amount_to_numeric(transactions)
    # Task 5 - Remove duplicate rows
    transactions = transactions.drop_duplicates()
    # ADDITIONAL CODE ADDED DUE TO COMPONENT TESTS FAILING
    # The expected and actual dataframes were not the same due
    # to index conflicts
    transactions.reset_index(drop=True, inplace=True)

    # Save the dataframe as a CSV for logging purposes
    # Ensure the directory exists
    save_dataframe_to_csv(transactions, OUTPUT_DIR, FILE_NAME)

    return transactions


def remove_missing_values(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.dropna(subset=["transaction_date"])

    # remove rows with null values in amount from the transaction dataframe
    transactions = transactions.dropna(subset=["amount"])

    return transactions


def standardise_date_format(transactions: pd.DataFrame) -> pd.DataFrame:
    # Convert all dates into dd/mm/yyyy format - write function to handle the
    # different types of date formats
    transactions["transaction_date"] = transactions["transaction_date"].apply(
        standardise_date  # type: ignore
    )
    transactions["transaction_date"] = transactions[
        "transaction_date"
    ].dt.strftime("%d/%m/%Y")
    transactions = transactions.dropna(subset=["transaction_date"])
    return transactions


def convert_amount_to_numeric(transactions: pd.DataFrame) -> pd.DataFrame:
    # Convert and drop NaNs in one operation
    transactions["amount"] = pd.to_numeric(
        transactions["amount"], errors="coerce"
    )
    transactions.dropna(subset=["amount"], inplace=True)

    return transactions
