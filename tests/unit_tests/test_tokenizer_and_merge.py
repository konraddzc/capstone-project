import pytest
import pandas as pd
from pathlib import Path
import pickle
from src.transform import transform


def test_build_merged_and_tokenizer(monkeypatch, tmp_path):
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(transform, "PROCESSED_DIR", processed_dir)

    df_fake_true = pd.DataFrame(
        {"text": ["fake news one", "real news one"], "label": [0, 1]}
    )
    df_welfake = pd.DataFrame({"text": ["other fake"], "label": [0]})
    df_ainews = pd.DataFrame({"text": ["ai real"], "label": [1]})

    transform.build_merged_and_tokenizer(df_fake_true, df_welfake, df_ainews)

    merged_path = processed_dir / "news_merged_clean.csv"
    tokenizer_path = processed_dir / "tokenizer_merged.pkl"

    assert merged_path.exists()
    assert tokenizer_path.exists()

    df_merged = pd.read_csv(merged_path)
    assert set(df_merged.columns) == {"text", "label"}
    assert len(df_merged) == 4

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    vocab = tokenizer.word_index
    for token in ["fake", "real", "news", "ai", "other"]:
        assert token in vocab
