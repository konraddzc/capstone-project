from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dropout,
    GlobalMaxPooling1D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_FILE = PROCESSED_DIR / "tokenizer_merged.pkl"

MAXLEN = 200
EPOCHS_PER_DATASET = 5
BATCH_SIZE = 64
TEST_SIZE = 0.2
LEARNING_RATE = 1e-4


def load_tokenizer():
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Expected tokenizer file at {TOKENIZER_FILE}. Run transform stage first."
        )
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def texts_to_X(texts, tokenizer, maxlen):
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen)
    return X


def build_model(vocab_size: int, maxlen: int = MAXLEN) -> Model:
    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size + 1, 100)(inp)
    x = Dropout(0.5)(x)
    x = LSTM(150, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset(csv_name: str, tokenizer):
    path = PROCESSED_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {csv_name} in {PROCESSED_DIR}. Run transform stage first."
        )
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).values
    X = texts_to_X(texts, tokenizer, MAXLEN)
    y = to_categorical(labels, num_classes=2)
    return X, y


def main():
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index)
    model = build_model(vocab_size=vocab_size, maxlen=MAXLEN)

    datasets = [
        ("fake_or_real_news_clean.csv", "stage1_fake_real"),
        ("ainews_clean.csv", "stage2_ainews"),
        ("welfake_clean.csv", "stage3_welfake"),
    ]

    for csv_name, stage_name in datasets:
        X, y = load_dataset(csv_name, tokenizer)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        print(f"Training on {csv_name} ({stage_name})")
        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS_PER_DATASET,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=1,
        )
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"{stage_name} validation accuracy: {acc:.4f}")

    final_model_path = MODELS_DIR / "iterative_model.keras"
    model.save(final_model_path)
    print(f"Saved iterative model to {final_model_path}")


if __name__ == "__main__":
    main()
