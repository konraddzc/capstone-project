import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import patch
from src.utils.file_utils import find_project_root, save_dataframe_to_csv


# Classes create suites inside a test file
class TestFindProjectRoot:
    def test_find_project_root_success(self):
        """Test finding project root when marker file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            nested_dir = os.path.join(temp_dir, "src", "utils")
            os.makedirs(nested_dir)

            # Create marker file in root
            marker_file = os.path.join(temp_dir, "README.md")
            with open(marker_file, "w") as f:
                f.write("test")

            # Mock __file__ to point to nested directory
            with patch(
                "src.utils.file_utils.__file__",
                os.path.join(nested_dir, "file_utils.py"),
            ):
                result = find_project_root("README.md")
                assert result == temp_dir

    def test_find_project_root_not_found(self):
        """Test FileNotFoundError when marker file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "src", "utils")
            os.makedirs(nested_dir)

            with patch(
                "src.utils.file_utils.__file__",
                os.path.join(nested_dir, "file_utils.py"),
            ):
                with pytest.raises(
                    FileNotFoundError,
                    match="Marker file 'nonexistent.txt' not found",
                ):
                    find_project_root("nonexistent.txt")

    def test_find_project_root_custom_marker(self):
        """Test finding project root with custom marker file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "src")
            os.makedirs(nested_dir)

            # Create custom marker file
            marker_file = os.path.join(temp_dir, "pyproject.toml")
            with open(marker_file, "w") as f:
                f.write("test")

            with patch(
                "src.utils.file_utils.__file__",
                os.path.join(nested_dir, "file_utils.py"),
            ):
                result = find_project_root("pyproject.toml")
                assert result == temp_dir


class TestSaveDataframeToCSV:
    def test_save_dataframe_to_csv_success(self):
        """Test successful DataFrame save to CSV."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.utils.file_utils.ROOT_DIR", temp_dir):
                save_dataframe_to_csv(df, "test_output", "test.csv")

                # Verify file was created
                expected_path = os.path.join(
                    temp_dir, "test_output", "test.csv"
                )
                assert os.path.exists(expected_path)

                # Verify content
                saved_df = pd.read_csv(expected_path)
                pd.testing.assert_frame_equal(df, saved_df)

    def test_save_dataframe_creates_directory(self):
        """
        Test that save_dataframe_to_csv creates output directory if it
        doesn't exist.
        """
        df = pd.DataFrame({"test": [1]})

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.utils.file_utils.ROOT_DIR", temp_dir):
                save_dataframe_to_csv(df, "new_dir/subdir", "test.csv")

                # Verify directory was created
                expected_dir = os.path.join(temp_dir, "new_dir", "subdir")
                assert os.path.exists(expected_dir)

                # Verify file exists
                expected_file = os.path.join(expected_dir, "test.csv")
                assert os.path.exists(expected_file)

    @patch("builtins.print")
    def test_save_dataframe_prints_confirmation(self, mock_print):
        """Test that save_dataframe_to_csv prints confirmation message."""
        df = pd.DataFrame({"test": [1]})

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.utils.file_utils.ROOT_DIR", temp_dir):
                save_dataframe_to_csv(df, "test_dir", "test.csv")

                expected_path = os.path.join(temp_dir, "test_dir", "test.csv")
                mock_print.assert_called_once_with(
                    f"Data saved to {expected_path}"
                )
