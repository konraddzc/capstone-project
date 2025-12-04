import pytest
from pathlib import Path
from src.extract import extract


@pytest.mark.parametrize(
    "dummy_dataset_name",
    ["test-dataset"],
)
def test_extract_creates_raw_directory(
    monkeypatch, tmp_path, dummy_dataset_name
):
    tmp_raw = tmp_path / "data" / "raw"
    monkeypatch.setattr(extract, "RAW_ROOT", tmp_raw)

    def fake_download(name):
        p = tmp_path / "kaggle_download" / name
        p.mkdir(parents=True)
        return str(p)

    monkeypatch.setattr(extract.kagglehub, "dataset_download", fake_download)

    extract.main()

    assert tmp_raw.exists()
    assert tmp_raw.is_dir()
