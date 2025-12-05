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
    raw_dir = tmp_path / "data" / "raw"
    monkeypatch.setattr(extract, "RAW_DIR", raw_dir)

    def fake_download(name):
        p = tmp_path / "kaggle_cache" / name / "versions" / "1"
        p.mkdir(parents=True, exist_ok=True)

        if "fake-and-real-news-dataset" in name:
            (p / "Fake.csv").write_text("text\nfake1\n", encoding="utf-8")
            (p / "True.csv").write_text("text\nreal1\n", encoding="utf-8")
        elif "fake-news-classification" in name:
            (p / "WELFake_Dataset.csv").write_text(
                "text,label\nx,1\n", encoding="utf-8"
            )
        elif "ai-fake-news-prediction" in name:
            (p / "AI_news.csv").write_text(
                "NEWS,FAKE\\REAL\nx,real\n", encoding="utf-8"
            )

        return str(p)

    monkeypatch.setattr(extract.kagglehub, "dataset_download", fake_download)

    extract.main()

    assert raw_dir.exists()
    assert raw_dir.is_dir()
