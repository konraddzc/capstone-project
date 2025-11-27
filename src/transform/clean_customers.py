import pandas as pd
from src.utils.file_utils import save_dataframe_to_csv

OUTPUT_DIR = "data/processed"
FILE_NAME = "cleaned_customers.csv"


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    # Task 2 - Remove rows with missing values
    customers = remove_missing_values(customers)
    # Task 3 - Remove the 'age' column
    customers = remove_age_column(customers)
    # Task 4 - Standardise the is_active column
    customers = standardise_is_active_column(customers)
    # Task 5 - Remove duplicate rows
    customers = customers.drop_duplicates()
    # ADDITIONAL CODE ADDED DUE TO COMPONENT TESTS FAILING
    # The expected and actual dataframes were not the same due
    # to index conflicts
    customers.reset_index(drop=True, inplace=True)

    # Save the dataframe as a CSV for logging purposes
    # Ensure the directory exists
    save_dataframe_to_csv(customers, OUTPUT_DIR, FILE_NAME)

    return customers


def remove_missing_values(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.dropna(subset=["country", "is_active"])


def remove_age_column(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.drop(columns=["age"])


def standardise_is_active_column(customers: pd.DataFrame) -> pd.DataFrame:
    # Convert active, 1 and true to True
    # Convert all other values to False
    # standardise is_active
    mapping = {
        "1": True,
        "0": False,
        "active": True,
        "inactive": False,
        "False": False,
        "True": True,
    }

    # Fill nan with False first, then convert to bool
    customers["is_active"] = (
        customers["is_active"]
        .map(mapping)
        .fillna(False)
        .infer_objects()
        .astype(bool)
    )
    return customers
