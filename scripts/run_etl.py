import os
import sys
from pathlib import Path
from config.env_config import setup_env
from src.extract.extract import main as extract_main
from src.transform.transform import main as transform_main
from src.utils.logging_utils import setup_logger


def main():
    logger = setup_logger("etl_pipeline", "etl_pipeline.log")
    for h in logger.handlers:
        print("DEBUG handler:", type(h), getattr(h, "baseFilename", None))
    try:
        logger.info("Starting ETL pipeline")

        logger.info("Beginning data extraction phase")
        extract_main()
        logger.info("Data extraction phase completed")

        logger.info("Beginning data transformation phase")
        transform_main()
        logger.info("Data transformation phase completed")

        logger.info("ETL pipeline completed successfully")
        print(
            f"ETL pipeline run successfully in {os.getenv('ENV', 'error')} environment!"
        )

        logger.info("Beginning data load phase")
        logger.info("Data load phase completed")

    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
