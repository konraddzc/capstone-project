import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
import nltk

st.set_page_config(page_title="Model Analysis", layout="wide")

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stMetric > div > div {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

TOKENIZER_PATH_DEFAULT = "notebooks/tokenizer.pkl"
TOKENIZER_PATH_ITER = "notebooks/tokenizer_iterative.pkl"
TOKENIZER_PATH_MERGED = "notebooks/tokenizer_merged.pkl"
MAXLEN = 200

DATASET_PATHS = {
    "Fake or Real Dataset": "data/processed/fake_or_real_news_clean.csv",
    "WELfake Dataset": "data/processed/welfake_clean.csv",
    "AI news Dataset": "data/processed/ainews_clean.csv",
    "Merged Dataset": "data/processed/news_merged_clean.csv",
}


@st.cache_resource
def load_selected_model(choice: str):
    if choice == "1.0":
        model_path = "notebooks/best_model.keras"
        tok_path = TOKENIZER_PATH_DEFAULT
    elif choice == "1.1":
        model_path = "notebooks/finetuned_model.keras"
        tok_path = TOKENIZER_PATH_DEFAULT
    elif choice == "1.2":
        model_path = "notebooks/finetuned_model.keras"
        tok_path = TOKENIZER_PATH_DEFAULT
    elif choice == "2.0":
        model_path = "notebooks/curriculum_best_model.keras"
        tok_path = TOKENIZER_PATH_ITER
    elif choice == "3.0":
        model_path = "notebooks/merged_model.keras"
        tok_path = TOKENIZER_PATH_MERGED
    else:
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    label_col = None
    for cand in ["label", "target", "y", "class"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(
            "Dataset must contain a label column named 'label', 'target', 'y' or 'class'."
        )

    y_raw = df[label_col]

    if y_raw.dtype == object:
        lowered = y_raw.astype(str).str.lower()
        uniq = set(lowered.unique())
        if uniq <= {"fake", "real"}:
            mapping = {"fake": 0, "real": 1}
            y = lowered.map(mapping)
        elif uniq <= {"0", "1"}:
            mapping = {"0": 0, "1": 1}
            y = lowered.map(mapping)
        else:
            raise ValueError(f"Unsupported string labels found: {uniq}")
    else:
        y = y_raw.astype(int)

    if y.isna().any():
        raise ValueError("Could not map all labels to numeric 0/1.")

    texts = df["text"].astype(str).tolist()
    return texts, y.to_numpy()


def process_text(text: str):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw and len(w) > 3]
    unique = np.unique(words, return_index=True)[1]
    return np.array(words)[np.sort(unique)].tolist()


def predict_on_texts(model, tokenizer, texts):
    processed = [" ".join(process_text(t)) for t in texts]
    seq = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(seq, maxlen=MAXLEN)
    preds = model.predict(padded, verbose=0)
    return preds


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def plot_roc(y_true, y_proba, positive_label=1):
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title("ROC curve", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    return fig


st.title("Model Analysis")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Model version", ["1.0", "1.1", "1.2", "2.0", "3.0"]
    )
    dataset_choice = st.selectbox("Dataset", list(DATASET_PATHS.keys()))

    sample_size = st.slider("Eval subset size", 200, 5000, 1000, 100)
    seed = st.number_input("Random seed", value=42, step=1)
    st.caption(
        "Subset is sampled uniformly at random (without replacement) "
        "using the seed above."
    )


model, tokenizer = load_selected_model(model_choice)
DATA_PATH = DATASET_PATHS[dataset_choice]

try:
    texts, y_true = load_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

n_total = len(texts)
n_eval = min(sample_size, n_total)

rng = np.random.default_rng(int(seed))
indices = rng.choice(n_total, size=n_eval, replace=False)

texts_eval = [texts[i] for i in indices]
y_true_eval = y_true[indices]

st.markdown(
    f"**Dataset:** `{dataset_choice}`  \n"
    f"**Total samples:** {n_total}  \n"
    f"**Evaluating on:** {n_eval} randomly sampled rows (seed = {int(seed)})"
)

run_eval = st.button(
    "Run evaluation", type="primary", use_container_width=True
)

if run_eval:
    if model is None or tokenizer is None:
        st.error("Could not load the selected model or tokenizer.")
    else:
        with st.spinner("Running model and computing metrics..."):
            try:
                probs = predict_on_texts(model, tokenizer, texts_eval)
                y_pred = np.argmax(probs, axis=1)
                y_proba_real = probs[:, 1]

                acc = accuracy_score(y_true_eval, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_eval, y_pred, average="binary", pos_label=1
                )

                cm = confusion_matrix(y_true_eval, y_pred, labels=[0, 1])
                class_names = ["fake", "real"]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc*100:.1f}%")
                col2.metric("Precision (real)", f"{precision*100:.1f}%")
                col3.metric("Recall (real)", f"{recall*100:.1f}%")
                col4.metric("F1 (real)", f"{f1*100:.1f}%")

                left_col, right_col = st.columns(2)

                with left_col:
                    st.subheader("Confusion matrix")
                    fig_cm = plot_confusion_matrix(cm, class_names)
                    st.pyplot(
                        fig_cm, clear_figure=True, use_container_width=False
                    )

                with right_col:
                    st.subheader("ROC curve")
                    fig_roc = plot_roc(
                        y_true_eval, y_proba_real, positive_label=1
                    )
                    st.pyplot(
                        fig_roc, clear_figure=True, use_container_width=False
                    )

                st.subheader("Classification report")
                report_dict = classification_report(
                    y_true_eval,
                    y_pred,
                    target_names=class_names,
                    output_dict=True,
                )
                report_df = (
                    pd.DataFrame(report_dict)
                    .T.rename_axis("class")
                    .reset_index()
                )
                report_df = report_df[
                    ["class", "precision", "recall", "f1-score", "support"]
                ]
                st.dataframe(report_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error during evaluation: {e}")
else:
    st.info(
        "Configure the model, dataset, subset size, and seed, then click **Run evaluation**."
    )
