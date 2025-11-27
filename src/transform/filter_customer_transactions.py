import pandas as pd


def filter_for_high_value_customers(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the merged data to return only high value customers.

    Args:
        merged_data (pd.DataFrame): The DataFrame containing merged customer
        and transaction data.

    Returns:
        pd.DataFrame: A DataFrame containing only high value customers.
    """
    # Define a threshold for high value customers
    threshold = 500  # Adjust as needed

    # Find the customers with total spend above the threshold
    customers_total_spend = find_high_value_total_spend(merged_data, threshold)
    # Calculate the average transaction amount for each customer
    customers_avg_transaction = find_avg_transaction_amount(merged_data)
    # Merge the two DataFrames to get the final high value customers
    high_value_customers = get_high_value_customers(
        customers_total_spend, customers_avg_transaction
    )
    return high_value_customers


def find_high_value_total_spend(
    merged_data: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    Finds customers with total spend above a certain threshold.

    Args:
        merged_data (pd.DataFrame): The DataFrame containing merged customer
        and transaction data.
        threshold (float): The minimum total spend to qualify as a high value
        customer.

    Returns:
        pd.DataFrame: A DataFrame of customers who meet the total spend
        criteria.
    """
    # Calculate the total amount spent by each customer
    total_spent = (
        merged_data.groupby("customer_id")["amount"].sum().reset_index()
    )

    # Rename the column to reflect the total spent
    total_spent.rename(columns={"amount": "total_spent"}, inplace=True)
    # Query the DataFrame for customers who have spent more than 500
    total_spent = total_spent.query("total_spent > 500")

    return total_spent


def find_avg_transaction_amount(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average transaction amount for each customer.

    Args:
        merged_data (pd.DataFrame): The DataFrame containing merged customer
        and transaction data.

    Returns:
        pd.DataFrame: A DataFrame with customer IDs and their average
        transaction amounts.
    """
    # Calculate the average transaction value for each customer
    avg_transaction_value = (
        merged_data.groupby("customer_id")["amount"].mean().reset_index()
    )

    # Rename the column to reflect the average transaction value
    avg_transaction_value.rename(
        columns={"amount": "avg_transaction_value"}, inplace=True
    )

    # Return the result
    return avg_transaction_value


def get_high_value_customers(
    total_spend: pd.DataFrame, avg_transaction: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges total spend and average transaction amount DataFrames to get high
    value customers.

    Args:
        total_spend (pd.DataFrame): DataFrame of customers with total spend
        above the threshold.
        avg_transaction (pd.DataFrame): DataFrame of customers with their
        average transaction amounts.

    Returns:
        pd.DataFrame: A DataFrame containing high value customers with their
        total spend and average transaction amount.
    """
    high_value_customers = pd.merge(
        total_spend, avg_transaction, on="customer_id"
    )
    return high_value_customers
