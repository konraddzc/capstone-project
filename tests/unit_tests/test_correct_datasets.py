import pytest
from pathlib import Path
from src.extract import extract


@pytest.mark.parametrize(
    "expected_datasets",
    [
        {
            "clmentbisaillon/fake-and-real-news-dataset",
            "saurabhshahane/fake-news-classification",
            "asminshan/ai-fake-news-prediction",
        }
    ],
)
def test_extract_calls_expected_kaggle_datasets(
    monkeypatch, tmp_path, expected_datasets
):
    raw_dir = tmp_path / "data" / "raw"
    monkeypatch.setattr(extract, "RAW_DIR", raw_dir)

    called = []

    def fake_download(name):
        called.append(name)
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

    assert set(called) == expected_datasets
