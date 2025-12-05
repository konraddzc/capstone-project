import pytest
import pandas as pd
from src.transform import transform


def test_transform_ainews_unit(monkeypatch, tmp_path):
    df_in = pd.DataFrame(
        {
            "NEWS": ["This is REAL news", "This is FAKE news"],
            "FAKE\\REAL": ["real", "fake"],
        }
    )

    ainews_csv = tmp_path / "AI_news.csv"
    df_in.to_csv(ainews_csv, index=False)

    monkeypatch.setattr(transform, "find_source", lambda name: ainews_csv)
    monkeypatch.setattr(transform, "PROCESSED_DIR", tmp_path)

    df_out = transform.transform_ainews().reset_index(drop=True)

    expected_text = [transform.process_text(t) for t in df_in["NEWS"]]
    expected_labels = [1, 0]

    assert df_out["label"].tolist() == expected_labels
    assert df_out["text"].tolist() == expected_text
