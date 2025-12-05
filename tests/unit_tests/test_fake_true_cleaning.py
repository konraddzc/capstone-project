import pytest
import pandas as pd
from src.transform import transform


@pytest.mark.parametrize(
    "fake_rows,true_rows",
    [
        (
            [
                {
                    "title": "t1",
                    "subject": "s1",
                    "date": "2021-01-01",
                    "text": "Fake news one",
                },
                {
                    "title": "t2",
                    "subject": "s2",
                    "date": "2021-01-02",
                    "text": "Fake news two",
                },
            ],
            [
                {
                    "title": "t3",
                    "subject": "s3",
                    "date": "2021-01-03",
                    "text": "Real news one",
                },
                {
                    "title": "t4",
                    "subject": "s4",
                    "date": "2021-01-04",
                    "text": "Real news two",
                },
            ],
        )
    ],
)
def test_transform_fake_true_unit(monkeypatch, tmp_path, fake_rows, true_rows):
    fake_df = pd.DataFrame(fake_rows)
    true_df = pd.DataFrame(true_rows)

    fake_csv = tmp_path / "Fake.csv"
    true_csv = tmp_path / "True.csv"
    fake_df.to_csv(fake_csv, index=False)
    true_df.to_csv(true_csv, index=False)

    def fake_find_source(name):
        if name == "Fake.csv":
            return fake_csv
        if name == "True.csv":
            return true_csv
        raise FileNotFoundError(name)

    monkeypatch.setattr(transform, "find_source", fake_find_source)
    monkeypatch.setattr(transform, "PROCESSED_DIR", tmp_path)

    df_out = transform.transform_fake_true().reset_index(drop=True)

    assert set(df_out.columns) == {"text", "label"}
    assert sorted(df_out["label"].tolist()) == [0, 0, 1, 1]
    assert df_out["text"].notna().all()
