import pytest
import pandas as pd
from src.transform import transform


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"title": "a", "text": "Fake row", "label": 1},
            {"title": "b", "text": "Real row", "label": 0},
        ]
    ],
)
def test_transform_welfake_unit(monkeypatch, tmp_path, rows):
    df_in = pd.DataFrame(rows)

    welfake_csv = tmp_path / "WELFake_Dataset.csv"
    df_in.to_csv(welfake_csv, index=False)

    monkeypatch.setattr(transform, "find_source", lambda name: welfake_csv)
    monkeypatch.setattr(transform, "PROCESSED_DIR", tmp_path)

    df_out = transform.transform_welfake().reset_index(drop=True)

    assert set(df_out.columns) == {"text", "label"}
    assert sorted(df_out["label"].tolist()) == [0, 1]
    assert df_out["text"].notna().all()
