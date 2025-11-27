# Friday Afternoon Task

> "Houston, we have a problem..."

You've been away on a training course, something to do with AWS!  Whilst you were away, the team continued to work on the Jupyter notebooks but at lunchtime they went out together and haven't returned.  Apparently, there was a booking mix up and the team is now doing some inpromptu tech support for the restaurant on their online booking system after one of them said, off-handedly:

> "It's probably a simple fix, can we get free food if we sort it?

...They're probably going to be "a while"...

---

## Current State

The project now has the complete Jupyter notebooks, some skeleton code for the actual implementation in the pipeline and - because the team is working to best practices - a full set of tests for the transaction stage of the process (to go along with the tests for the extract phase).

They also completed some utility functions and tests, some of which were taken from the Jupyter Notebooks.

As you can guess, there's a lot of failing tests!

### Failing Unit Tests

```sh
FAILED tests/unit_tests/test_clean_customers.py::TestRemoveMissingValues::test_remove_missing_values_drops_rows_with_missing_country - assert 0 == 2
FAILED tests/unit_tests/test_clean_customers.py::TestRemoveMissingValues::test_remove_missing_values_drops_rows_with_missing_is_active - assert 0 == 2
FAILED tests/unit_tests/test_clean_customers.py::TestRemoveMissingValues::test_remove_missing_values_keeps_complete_rows - assert 0 == 2
FAILED tests/unit_tests/test_clean_customers.py::TestRemoveAgeColumn::test_remove_age_column_drops_age_column - AssertionError: assert [] == ['customer_id...e', 'country']
FAILED tests/unit_tests/test_clean_customers.py::TestRemoveAgeColumn::test_remove_age_column_preserves_other_columns - assert 0 == 2
FAILED tests/unit_tests/test_clean_customers.py::TestStandardiseIsActiveColumn::test_standardise_is_active_column_converts_values - KeyError: 'is_active'
FAILED tests/unit_tests/test_clean_customers.py::TestStandardiseIsActiveColumn::test_standardise_is_active_column_preserves_other_columns - KeyError: 'customer_id'
FAILED tests/unit_tests/test_clean_customers.py::TestCleanCustomers::test_clean_customers_full_pipeline - assert 0 == 3
FAILED tests/unit_tests/test_clean_transactions.py::TestRemoveMissingValues::test_remove_missing_values_success - assert 0 == 2
FAILED tests/unit_tests/test_clean_transactions.py::TestRemoveMissingValues::test_remove_missing_values_no_missing - assert 0 == 2
FAILED tests/unit_tests/test_clean_transactions.py::TestConvertAmountToNumeric::test_convert_amount_to_numeric_success - KeyError: 'amount'
FAILED tests/unit_tests/test_clean_transactions.py::TestConvertAmountToNumeric::test_convert_amount_to_numeric_with_invalid - assert 0 == 2
FAILED tests/unit_tests/test_merge_customers_transactions.py::TestMergeTransactionsCustomers::test_successful_merge - assert 0 == 3
FAILED tests/unit_tests/test_merge_customers_transactions.py::TestMergeTransactionsCustomers::test_partial_match - assert 0 == 2
FAILED tests/unit_tests/test_merge_customers_transactions.py::TestMergeTransactionsCustomers::test_empty_dataframes - assert 0 == 7
FAILED tests/unit_tests/test_merge_customers_transactions.py::TestMergeTransactionsCustomers::test_duplicate_transactions - assert 0 == 3
ERROR tests/unit_tests/test_filter_customer_transactions.py::test_filter_for_high_value_customers - FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data/expected_merged_clean_results.csv'
ERROR tests/unit_tests/test_filter_customer_transactions.py::test_find_high_value_total_spend - FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data/expected_merged_clean_results.csv'
ERROR tests/unit_tests/test_filter_customer_transactions.py::test_find_avg_transaction_amount - FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data/expected_merged_clean_results.csv'
```

### Failing Component Tests

```sh
FAILED tests/component_tests/test_transform_customers_component.py::test_transform_customers_handles_empty_dataframe - Failed: DID NOT RAISE <class 'KeyError'>
FAILED tests/component_tests/test_transform_customers_component.py::test_transform_customers_missing_required_column - Failed: DID NOT RAISE <class 'KeyError'>
FAILED tests/component_tests/test_transform_transactions_component.py::test_transform_transactions_handles_empty_dataframe - Failed: DID NOT RAISE <class 'KeyError'>
FAILED tests/component_tests/test_transform_transactions_component.py::test_transform_transactions_handles_empty_rows_with_columns - Failed: DID NOT RAISE <class 'AttributeError'>
FAILED tests/component_tests/test_transform_transactions_component.py::test_transform_transactions_handles_all_rows_filtered - Failed: DID NOT RAISE <class 'AttributeError'>
FAILED tests/component_tests/test_transform_transactions_component.py::test_transform_transactions_missing_required_column - Failed: DID NOT RAISE <class 'KeyError'>
ERROR tests/component_tests/test_transform_customers_component.py::test_transform_customers_returns_expected_data - FileNotFoundError: [Errno 2] No such file or directory: '/Users/edwright/DFA-Repos/de-2509-a/etl-project-walkthrough/tests/component_tests/../test_data/expected_customers_clean_...
ERROR tests/component_tests/test_transform_data.py::test_transform_data_returns_correct_structure - FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data/expected_merged_clean_results.csv'
ERROR tests/component_tests/test_transform_transactions_component.py::test_transform_transactions_returns_expected_data - FileNotFoundError: [Errno 2] No such file or directory: '/Users/edwright/DFA-Repos/de-2509-a/etl-project-walkthrough/tests/component_tests/../test_data/expected_transactions_cle...
```

### Failing Integration Tests

```sh
FAILED tests/integration_tests/test_extract_transform_integration.py::test_extract_transform_integration - KeyError: 'customer_id'
FAILED tests/integration_tests/test_extract_transform_integration.py::test_extract_transform_error_handling - AssertionError: Regex pattern did not match.
FAILED tests/integration_tests/test_extract_transform_integration.py::test_extract_transform_performance - KeyError: 'customer_id'
FAILED tests/integration_tests/test_extract_transform_integration.py::test_extract_transform_data_volume - KeyError: 'customer_id'
```

### Failing e2e Tests

```sh
FAILED tests/e2e_tests/test_etl_pipeline.py::test_full_etl_pipeline_success - AssertionError: ETL pipeline failed: 2025-11-14 12:32:03,820 - etl_pipeline - INFO - Starting ETL pipeline
```

---
---

## YOUR TASK

Your task is to fix the code so that all tests pass successfully!
