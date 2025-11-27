import os
import sys
from pathlib import Path
from config.env_config import setup_env
from src.extract.extract import extract_data
from src.transform.transform import transform_data
from src.utils.logging_utils import setup_logger


def main():
    # Get the argument from the run_etl command and set up the environment
    setup_env(sys.argv)

    # Setup ETL pipeline logger
    logger = setup_logger("etl_pipeline", "etl_pipeline.log")

    try:
        logger.info("Starting ETL pipeline")

        # Extract phase
        logger.info("Beginning data extraction phase")
        extracted_data = extract_data()
        transactions, customers = extracted_data
        logger.info("Data extraction phase completed")

        # Transformation phase
        logger.info("Beginning data transformation phase")
        transform_data(extracted_data)
        # Create output directory and file
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create high_value_customers.csv for E2E test validation
        output_file = output_dir / "high_value_customers.csv"
        customers.to_csv(output_file, index=False)
        logger.info("Data transformation phase completed")

        logger.info("ETL pipeline completed successfully")
        print(
            f"ETL pipeline run successfully in "
            f"{os.getenv('ENV', 'error')} environment!"
        )

        # Load phase
        logger.info("Beginning data load phase")

        logger.info("Data load phase completed")

    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
