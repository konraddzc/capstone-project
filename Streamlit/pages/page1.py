import re
import pickle
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
import pandas as pd
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

MODEL_PATH = "../notebooks/best_model.keras"
TOKENIZER_PATH = "../notebooks/tokenizer.pkl"
MAXLEN = 200

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(MODEL_PATH)
embedding_layer = model.layers[1]
emb_dim = embedding_layer.output_dim
emb_input = tf.keras.Input(shape=(MAXLEN, emb_dim))
x = emb_input
for layer in model.layers[2:]:
    x = layer(x)
rest_model = tf.keras.Model(emb_input, x)


def fetch_article_text(url, timeout=15):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    selectors = [
        "article",
        "main",
        "div[itemprop='articleBody']",
        "div#content",
        "div.article-body",
        "div.post-content",
        "section.article",
        "section#content",
        "div.story-body__inner",
    ]
    for sel in selectors:
        container = soup.select_one(sel)
        if container:
            paragraphs = container.find_all("p")
            texts = [
                p.get_text(" ", strip=True)
                for p in paragraphs
                if p.get_text(strip=True)
            ]
            if texts:
                return "\n".join(texts).strip()
    body = soup.body if soup.body else soup
    paragraphs = body.find_all("p")
    texts = [
        p.get_text(" ", strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    ]
    return "\n".join(texts).strip()


def process_text(text):
    text = re.sub(r"\s+", " ", text, flags=re.I)
    text = re.sub(r"\W", " ", str(text))
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]
    Words = [word for word in Words if len(word) > 3]
    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()
    return cleaned_text


def explain_url(url, top_k=20):
    text = fetch_article_text(url)
    tokens = process_text(text)
    if len(tokens) == 0:
        return {
            "prediction": "fake",
            "probabilities": [1.0, 0.0],
            "important_tokens": [],
        }
    seq = tokenizer.texts_to_sequences([" ".join(tokens)])
    seq = np.array(seq)
    padded = pad_sequences(seq, maxlen=MAXLEN)
    preds = model.predict(padded, verbose=0)
    pred_class = int(np.argmax(preds[0]))
    with tf.GradientTape() as tape:
        embeddings = embedding_layer(padded)
        tape.watch(embeddings)
        outputs = rest_model(embeddings)
        target = outputs[:, pred_class]
    grads = tape.gradient(target, embeddings)
    token_importance = tf.norm(grads, axis=-1).numpy()[0]
    seq_ids = padded[0]
    valid = seq_ids != 0
    token_ids = seq_ids[valid]
    token_scores = token_importance[valid]
    id2word = {i: w for w, i in tokenizer.word_index.items()}
    words = [id2word.get(int(t), "[UNK]") for t in token_ids]
    if len(token_scores) > 0:
        token_scores = token_scores / (token_scores.max() + 1e-8)
    ranked = sorted(zip(words, token_scores), key=lambda x: x[1], reverse=True)
    ranked = [(w, round(float(s), 3)) for w, s in ranked[:top_k]]
    label = "real" if pred_class == 1 else "fake"
    return {
        "prediction": label,
        "probabilities": preds[0].tolist(),
        "important_tokens": ranked,
    }


st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Fake News Detector")

url = st.text_input("Enter article URL")
top_k = st.slider(
    "Number of important tokens to display",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)

if st.button("Analyze") and url:
    try:
        result = explain_url(url, top_k=top_k)
        st.subheader("Prediction")
        st.write(f"Label: **{result['prediction'].upper()}**")
        probs = result["probabilities"]
        prob_fake = probs[0] if len(probs) > 0 else None
        prob_real = probs[1] if len(probs) > 1 else None
        if prob_fake is not None:
            st.write(f"Probability fake: {prob_fake:.4f}")
        if prob_real is not None:
            st.write(f"Probability real: {prob_real:.4f}")
        tokens = result["important_tokens"]
        if tokens:
            df_tokens = pd.DataFrame(tokens, columns=["token", "importance"])
            st.subheader("Most important tokens")
            st.dataframe(df_tokens)
        else:
            st.write("No tokens extracted from the article text.")
    except Exception as e:
        st.error(f"Error processing URL: {e}")
