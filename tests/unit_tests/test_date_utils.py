import pytest
import pandas as pd
from src.utils.date_utils import standardise_date


@pytest.mark.parametrize(
    "date_str, expected",
    [
        # %Y/%m/%d format
        ("2021/01/01", pd.Timestamp("2021-01-01")),
        # %Y-%m-%d format
        ("2021-03-01", pd.Timestamp("2021-03-01")),
        # %d %b %Y format
        ("01 Jan 2021", pd.Timestamp("2021-01-01")),
        ("15 Mar 2022", pd.Timestamp("2022-03-15")),
        # %b %d, %Y format
        ("Jan 01, 2021", pd.Timestamp("2021-01-01")),
        ("Dec 25, 2023", pd.Timestamp("2023-12-25")),
        # %d %B %Y format
        ("01 January 2021", pd.Timestamp("2021-01-01")),
        ("31 December 2022", pd.Timestamp("2022-12-31")),
        # %d-%m-%Y format
        ("01-02-2021", pd.Timestamp("2021-02-01")),
        ("25-12-2023", pd.Timestamp("2023-12-25")),
        # %d/%m/%Y format
        ("01/05/2021", pd.Timestamp("2021-05-01")),
        ("31/12/2022", pd.Timestamp("2022-12-31")),
        # %m/%d/%Y format
        ("05/01/2021", pd.Timestamp("2021-01-05")),
        ("12/25/2023", pd.Timestamp("2023-12-25")),
        # Invalid cases
        ("invalid date", pd.NaT),
        ("", pd.NaT),
        (None, pd.NaT),
        ("2021/13/01", pd.NaT),  # Invalid month
        ("2021/01/32", pd.NaT),  # Invalid day
        ("2021-02-30", pd.NaT),  # Invalid date
    ],
)
def test_standardise_date(date_str, expected):
    result = standardise_date(date_str)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected
