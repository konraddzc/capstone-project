import pytest
from pathlib import Path
from src.transform import transform


def test_find_source_raises_when_root_missing(monkeypatch, tmp_path):
    missing_root = tmp_path / "no_raw_here"
    monkeypatch.setattr(transform, "RAW_ROOT", missing_root)
    with pytest.raises(FileNotFoundError):
        transform.find_source("Fake.csv")


def test_find_source_raises_when_file_missing(monkeypatch, tmp_path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True)
    monkeypatch.setattr(transform, "RAW_ROOT", raw_root)
    with pytest.raises(FileNotFoundError):
        transform.find_source("Fake.csv")
