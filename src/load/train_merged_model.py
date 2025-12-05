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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = PROCESSED_DIR / "news_merged_clean.csv"
TOKENIZER_FILE = PROCESSED_DIR / "tokenizer_merged.pkl"

MAXLEN = 200
EPOCHS = 15
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


def build_model(vocab_size: int, maxlen: int = MAXLEN) -> tf.keras.Model:
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


def prepare_xy(df: pd.DataFrame, tokenizer, maxlen: int = MAXLEN):
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).values
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen)
    return X, labels


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Expected {DATA_FILE.name} in {PROCESSED_DIR}. Run transform stage first."
        )
    df = pd.read_csv(DATA_FILE)
    tokenizer = load_tokenizer()
    X, labels = prepare_xy(df, tokenizer, maxlen=MAXLEN)

    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
        X, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )

    y_train = to_categorical(y_train_labels, num_classes=2)
    y_test = to_categorical(y_test_labels, num_classes=2)

    vocab_size = len(tokenizer.word_index)
    model = build_model(vocab_size=vocab_size, maxlen=MAXLEN)

    checkpoint_path = MODELS_DIR / "merged_model_best.keras"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1,
    )

    final_model_path = MODELS_DIR / "merged_model.keras"
    model.save(final_model_path)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Merged model – test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
