import re
import pickle
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

TOKENIZER_PATH_DEFAULT = "notebooks/tokenizer.pkl"
TOKENIZER_PATH_ITER = "notebooks/tokenizer_iterative.pkl"
MAXLEN = 200

explainer = LimeTextExplainer(class_names=["fake", "real"])


@st.cache_resource
def load_selected_model(choice: str):
    if choice == "1.0":
        model_path = "notebooks/best_model.keras"
        tokenizer_path = TOKENIZER_PATH_DEFAULT
    elif choice == "1.1":
        model_path = "notebooks/finetuned_model.keras"
        tokenizer_path = TOKENIZER_PATH_DEFAULT
    elif choice == "1.2":
        model_path = "notebooks/v3_model.keras"
        tokenizer_path = TOKENIZER_PATH_DEFAULT
    elif choice == "2.0":
        model_path = "notebooks/curriculum_best_model.keras"
        tokenizer_path = TOKENIZER_PATH_ITER
    elif choice == "3.0":
        model_path = "notebooks/merged_model.keras"
        tokenizer_path = "notebooks/tokenizer_merged.pkl"
    else:
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_data(show_spinner=False)
def fetch_article_text(url, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    soup = BeautifulSoup(r.text, "html.parser")

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
        c = soup.select_one(sel)
        if c:
            ps = c.find_all("p")
            txts = [
                p.get_text(" ", strip=True)
                for p in ps
                if p.get_text(strip=True)
            ]
            if txts:
                return "\n\n".join(txts).strip()

    ps = soup.find_all("p")
    return "\n\n".join([p.get_text(" ", strip=True) for p in ps]).strip()


def process_text(text: str):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text.lower())
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(w) for w in tokens]
    sw = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in sw and len(w) > 3]
    uniq_idx = np.unique(tokens, return_index=True)[1]
    return np.array(tokens)[np.sort(uniq_idx)].tolist()


def predict_factory(model, tokenizer):
    def predict_fn(text_list):
        processed = [" ".join(process_text(t)) for t in text_list]
        seq = tokenizer.texts_to_sequences(processed)
        padded = pad_sequences(seq, maxlen=MAXLEN)
        p = model.predict(padded, verbose=0)
        p = np.array(p)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if p.shape[1] == 1:
            real = p
            fake = 1 - real
            p = np.concatenate([fake, real], axis=1)
        return p

    return predict_fn


def build_highlighted_html(exp, label_idx, text):
    weights = dict(exp.as_list(label=label_idx))
    weights = {k.lower(): v for k, v in weights.items()}
    max_abs = max(abs(v) for v in weights.values()) if weights else 1
    parts = re.split(r"(\W+)", text)
    spans = []

    for p in parts:
        if p == "":
            continue
        if re.match(r"\W+", p):
            spans.append(p)
            continue
        k = p.lower()
        w = weights.get(k, 0)
        if w != 0:
            alpha = 0.15 + 0.4 * (abs(w) / max_abs)
            if w > 0:
                color = f"rgba(70, 170, 70, {alpha:.2f})"
            else:
                color = f"rgba(200, 80, 80, {alpha:.2f})"
            spans.append(
                f"<span style='background:{color}; padding:0 3px; "
                f"border-radius:2px;'>{p}</span>"
            )
        else:
            spans.append(p)

    rendered = "".join(spans)
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{
            margin: 0;
            padding: 0;
            background: #111;
            font-family: Georgia, serif;
            color: #e0e0e0;
          }}
          .container {{
            width: 820px;
            margin: 2rem auto;
            font-size: 1.05rem;
            line-height: 1.68;
          }}
          h2 {{
            font-weight: 600;
            text-align: left;
            margin-bottom: 1.5rem;
            letter-spacing: 0.3px;
          }}
          .article {{
            white-space: pre-line;
            text-align: justify;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>Model Attribution Highlights</h2>
          <div class="article">{rendered}</div>
        </div>
      </body>
    </html>
    """
    return html


def explain(model, tokenizer, text, top_k):
    predict_fn = predict_factory(model, tokenizer)
    p = predict_fn([text])[0]
    label_idx = int(np.argmax(p))
    exp = explainer.explain_instance(
        text, predict_fn, num_features=top_k, labels=[0, 1]
    )
    amap = exp.as_map()
    if label_idx not in amap:
        label_idx = sorted(amap.keys())[0]
    label = "real" if np.argmax(p) == 1 else "fake"
    html = build_highlighted_html(exp, label_idx, text)
    return label, p.tolist(), html


st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detection: Model Interpretation")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Model version", ["1.0", "1.1", "1.2", "2.0", "3.0"]
    )
    input_mode = st.radio("Input source", ["URL", "Paste text"])
    top_k = st.slider("Highlighted tokens", 5, 50, 20, 5)

model, tokenizer = load_selected_model(model_choice)

if input_mode == "URL":
    url = st.text_input("Article URL")
    txt_manual = None
else:
    txt_manual = st.text_area("Article text", height=220)
    url = ""

run = st.button("Analyze", use_container_width=True)

if run:
    try:
        txt = None
        if input_mode == "URL":
            if url:
                txt = fetch_article_text(url)
                if not txt:
                    st.error("Unable to extract text.")
            else:
                st.warning("Enter a URL.")
        else:
            if txt_manual and txt_manual.strip():
                txt = txt_manual.strip()

        if txt:
            txt = txt.lstrip()
            label, probs, html = explain(model, tokenizer, txt, top_k)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Prediction")
                st.markdown(
                    f"""
                    <div style="
                        background:#1c1c1c;
                        padding:0.75rem 1rem;
                        border-radius:6px;
                        border:1px solid #333;
                    ">
                    <div style="font-size:0.85rem;color:#aaa;">Model version</div>
                    <div style="font-weight:600;">v{model_choice}</div>
                    <div style="margin-top:0.6rem;font-size:0.85rem;color:#aaa;">Classification</div>
                    <div style="font-size:1.15rem;font-weight:700;">
                    {label.upper()}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.subheader("Class probabilities")
                st.metric("Fake", f"{float(probs[0])*100:.1f}%")
                st.metric("Real", f"{float(probs[1])*100:.1f}%")

            st.subheader("Model Explanation")
            components.html(html, height=900, scrolling=True)

            with st.expander("Article text"):
                st.text_area("Raw text", txt, height=220)

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
