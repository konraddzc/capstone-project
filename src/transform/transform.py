from pathlib import Path
import re
import pickle

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def process_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def find_source(filename: str) -> Path:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"{RAW_ROOT} does not exist")
    matches = list(RAW_ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {RAW_ROOT}")
    return matches[0]


def transform_fake_true() -> pd.DataFrame:
    fake_path = find_source("Fake.csv")
    true_path = find_source("True.csv")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake = df_fake.assign(label=0)
    df_true = df_true.assign(label=1)
    for col in ["subject", "date", "title"]:
        if col in df_fake.columns:
            df_fake = df_fake.drop(columns=[col])
        if col in df_true.columns:
            df_true = df_true.drop(columns=[col])
    if "text" not in df_fake.columns or "text" not in df_true.columns:
        raise ValueError("Expected 'text' column in Fake.csv and True.csv.")
    df = pd.concat(
        [df_fake[["text", "label"]], df_true[["text", "label"]]],
        ignore_index=True,
    )
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(inplace=True)
    df["text"] = df["text"].astype(str).map(process_text)
    df["label"] = df["label"].astype(int)
    out_path = PROCESSED_DIR / "fake_or_real_news_clean.csv"
    df.to_csv(out_path, index=False)
    return df


def transform_welfake() -> pd.DataFrame:
    welfake_path = find_source("WELFake_Dataset.csv")
    df_welfake = pd.read_csv(welfake_path)
    if "label" not in df_welfake.columns:
        raise ValueError("Expected 'label' column in WELFake_Dataset.csv.")
    wel_label_map = {1: 0, 0: 1}
    df_welfake["label"] = (
        df_welfake["label"].map(wel_label_map).astype(int).values
    )
    if "title" in df_welfake.columns:
        df_welfake = df_welfake.drop(columns=["title"])
    if "text" not in df_welfake.columns:
        raise ValueError("Expected 'text' column in WELFake_Dataset.csv.")
    df = df_welfake[["text", "label"]]
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(inplace=True)
    df["text"] = df["text"].astype(str).map(process_text)
    df["label"] = df["label"].astype(int)
    out_path = PROCESSED_DIR / "welfake_clean.csv"
    df.to_csv(out_path, index=False)
    return df


def transform_ainews() -> pd.DataFrame:
    ainews_path = find_source("AI_news.csv")
    df_ainews = pd.read_csv(ainews_path)

    if "NEWS" not in df_ainews.columns:
        raise ValueError(
            f"Expected 'NEWS' column in AI_news.csv. Columns: {df_ainews.columns.tolist()}"
        )

    if "FAKE\\REAL" not in df_ainews.columns:
        raise ValueError(
            f"Expected 'FAKE\\REAL' column in AI_news.csv. Columns: {df_ainews.columns.tolist()}"
        )

    df_ainews.rename(
        columns={"NEWS": "text", "FAKE\\REAL": "label"}, inplace=True
    )

    ainews_label_map = {"real": 1, "fake": 0}
    df_ainews["label"] = df_ainews["label"].str.lower().map(ainews_label_map)

    df_ainews.dropna(subset=["label", "text"], inplace=True)
    df_ainews.drop_duplicates(inplace=True)

    df_ainews["label"] = df_ainews["label"].astype(int)
    df_ainews["text"] = df_ainews["text"].astype(str).map(process_text)

    out_path = PROCESSED_DIR / "ainews_clean.csv"
    df_ainews.to_csv(out_path, index=False)
    return df_ainews[["text", "label"]]


def build_merged_and_tokenizer(
    df_fake_true: pd.DataFrame,
    df_welfake: pd.DataFrame,
    df_ainews: pd.DataFrame,
):
    df_merged = pd.concat(
        [df_fake_true, df_welfake, df_ainews], ignore_index=True
    )
    df_merged.dropna(subset=["text"], inplace=True)
    df_merged.drop_duplicates(inplace=True)
    merged_path = PROCESSED_DIR / "news_merged_clean.csv"
    df_merged.to_csv(merged_path, index=False)
    texts = df_merged["text"].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    tokenizer_path = PROCESSED_DIR / "tokenizer_merged.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)


def main():
    df_fake_true = transform_fake_true()
    df_welfake = transform_welfake()
    df_ainews = transform_ainews()
    build_merged_and_tokenizer(df_fake_true, df_welfake, df_ainews)


if __name__ == "__main__":
    main()
