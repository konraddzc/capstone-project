import logging
from src.utils.logging_utils import setup_logger
from src.utils.db_utils import QueryExecutionError

logger = setup_logger(__name__, "database_query.log", level=logging.DEBUG)


def import_sql_query(filename: str) -> str:
    """Read SQL query from file and return as single-line string.

    Args:
        filename: Path to the SQL file to read.

    Returns:
        SQL query as a single-line string with newlines replaced by spaces.

    Raises:
        QueryExecutionError: If file is not found or cannot be read.
    """
    try:
        with open(filename, "r") as file:
            imported_query = file.read().replace("\n", " ").strip()
            logger.info(f"Successfully imported query from {filename}")
            return imported_query
    except FileNotFoundError as e:
        logger.error(f"Failed to import query: {filename} not found")
        raise QueryExecutionError(f"Failed to import query: {e}")
