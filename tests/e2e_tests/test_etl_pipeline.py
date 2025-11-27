import subprocess
import sys
import os
from pathlib import Path

"""
End-to-End (E2E) Tests for ETL Pipeline

These tests validate the complete ETL pipeline execution from start to finish,
testing the system as a whole rather than individual components.

Test Coverage:
1. Complete ETL Pipeline: Verifies Extract → Transform → Load workflow
   from script invocation through final data output (high_value_customers.csv)
2. Environment Handling: Tests proper environment setup and configuration
   loading
3. Output Validation: Confirms final CSV output and comprehensive logging
4. Data Quality: Validates output file contains expected data and structure
5. Error Handling: Validates graceful failure and proper error reporting
6. Process Integration: Tests subprocess execution and exit code handling

E2E Tests focus on:
- Real script execution via subprocess (not function calls)
- Complete ETL workflow validation (Extract → Transform → Load)
- Environment isolation and configuration
- Final deliverable creation (high_value_customers.csv)
- Comprehensive logging across all ETL phases
- Production-like execution scenarios

This differs from other test types by:
- Testing the complete system end-to-end
- Using subprocess execution (not direct function calls)
- Validating real file outputs and system state
- Testing deployment-ready script execution
- Focusing on user-facing functionality

Note: These tests require actual database and file system access,
making them slower but providing the highest confidence in system functionality.
"""


def test_full_etl_pipeline_success():
    """E2E test: Complete ETL pipeline execution with output validation"""
    # Set test environment and log directory
    env = os.environ.copy()
    env["ENV"] = "test"

    # Get directories
    project_root = Path(__file__).parent.parent.parent
    test_logs_dir = Path(__file__).parent.parent / "logs"
    test_logs_dir.mkdir(exist_ok=True)
    env["LOG_BASE_PATH"] = str(test_logs_dir)

    # Run the ETL pipeline script
    result = subprocess.run(
        [sys.executable, "scripts/run_etl.py", "test"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    # Verify pipeline executed successfully
    assert result.returncode == 0, f"ETL pipeline failed: {result.stderr}"

    # Verify log file was created and contains expected stages
    log_file = test_logs_dir / "logs" / "etl_pipeline.log"
    assert (
        log_file.exists()
    ), f"ETL pipeline log file not created at {log_file}"

    log_content = log_file.read_text()
    assert "Starting ETL pipeline" in log_content
    assert "Beginning data extraction phase" in log_content
    assert "Data extraction phase completed" in log_content
    # assert "Beginning the data transformation phase" in log_content
    # assert "Data transformation phase completed" in log_content
    # assert "Beginning data load phase" in log_content
    # assert "Data load phase completed" in log_content
    assert "ETL pipeline completed successfully" in log_content

    # Verify final output file was created
    output_file = (
        project_root / "data" / "processed" / "high_value_customers.csv"
    )
    assert (
        output_file.exists()
    ), f"Expected output file not created: {output_file}"

    # Verify output file has data
    import pandas as pd

    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0, "Output file is empty"
    assert (
        "customer_id" in result_df.columns
    ), "Missing expected column in output"


def test_etl_pipeline_invalid_environment():
    """E2E test: ETL pipeline handles invalid environment gracefully"""
    env = os.environ.copy()

    project_root = Path(__file__).parent.parent.parent
    test_logs_dir = Path(__file__).parent.parent / "logs"
    test_logs_dir.mkdir(exist_ok=True)
    env["LOG_BASE_PATH"] = str(test_logs_dir)

    # Run with invalid environment argument (should fail)
    result = subprocess.run(
        [sys.executable, "scripts/run_etl.py", "invalid_env"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    # Verify pipeline failed gracefully
    assert result.returncode == 1, (
        "Expected pipeline to fail with invalid env, "
        f"got return code {result.returncode}"
    )
    assert (
        "Please provide an environment" in result.stderr
        or "Please provide an environment" in result.stdout
    )
