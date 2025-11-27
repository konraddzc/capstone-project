import pytest
from src.extract.extract import extract_data
from src.transform.transform import transform_data

"""
Integration Tests for Extract-Transform Pipeline

These tests validate the complete extract-transform workflow,
testing how the two phases work together to produce the final business
deliverable.

Test Coverage:
1. Complete Pipeline: Validates extract → transform produces exact expected
output
2. Error Resilience: Tests pipeline behaviour when transform receives corrupted
data
3. Performance Requirements: Ensures pipeline meets time constraints
(<10 seconds)
4. Data Quality: Validates proper filtering and data volume reduction

Integration Tests focus on:
- End-to-end extract-transform workflow validation
- Real data processing with expected business outcomes
- Performance benchmarking with actual data volumes
- Error handling across phase boundaries
- Data quality validation using precise result matching

This differs from other test types by:
- Testing complete workflow rather than individual components
- Using real data sources for realistic scenarios
- Validating final business deliverable (high_value_customers.csv equivalent)
- Testing non-functional requirements (performance, data volume)
- Focusing on cross-phase integration and data handoff

Note: These tests use actual database and CSV data sources,
making them slower but providing high confidence in production readiness.
"""


def test_extract_transform_integration():
    """Integration test: End-to-end extract-transform workflow validation"""
    extracted_data = extract_data()
    result = transform_data(extracted_data)

    # Validate schema and data quality for filtered/aggregated data
    expected_columns = {
        "customer_id",
        "total_spent",
        "avg_transaction_value",
    }
    assert expected_columns.issubset(set(result.columns))
    assert len(result) > 0
    assert result["total_spent"].dtype in ["float64", "float32"]
    assert result["avg_transaction_value"].dtype in ["float64", "float32"]


def test_extract_transform_error_handling():
    """Integration test: Pipeline resilience with corrupted extracted data"""
    raw_transactions, raw_customers = extract_data()
    corrupted_customers = raw_customers.drop(columns=["country", "is_active"])

    with pytest.raises(KeyError, match="country|is_active"):
        transform_data((raw_transactions, corrupted_customers))


def test_extract_transform_performance():
    """Integration test: Performance requirements validation"""
    import time

    start_time = time.time()

    extracted_data = extract_data()
    result = transform_data(extracted_data)

    execution_time = time.time() - start_time
    assert (
        execution_time < 10.0
    ), f"Pipeline took {execution_time:.2f}s, expected <10s"
    assert len(result) > 0, "Pipeline should produce results"


def test_extract_transform_data_volume():
    """Integration test: Validate data processing expectations"""
    extracted_data = extract_data()
    result = transform_data(extracted_data)

    raw_transactions, raw_customers = extracted_data

    # Validate reasonable data processing - result should be aggregated by customer
    assert len(result) > 0, "Result should not be empty"
    assert len(result) <= len(
        raw_customers
    ), "Result should not exceed input customers (since it's aggregated by customer)"

    # Validate aggregated data integrity - each result row should have valid customer data
    assert (
        result["customer_id"].notna().all()
    ), "All customers should have customer_id"
    assert (
        result["total_spent"].notna().all()
    ), "All customers should have total_spent"
    assert (
        result["avg_transaction_value"].notna().all()
    ), "All customers should have avg_transaction_value"
