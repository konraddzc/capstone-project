import re
import io
import pickle
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from bs4 import BeautifulSoup
from urllib.parse import urlparse
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

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path, compile=False)
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

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    domain = urlparse(url).netloc
    mode = "none"
    paragraph_count = 0

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
                paragraph_count = len(txts)
                mode = "structured"
                text = "\n\n".join(txts).strip()
                meta = {
                    "paragraphs": paragraph_count,
                    "title": title,
                    "domain": domain,
                    "mode": mode,
                }
                return text, meta

    ps = soup.find_all("p")
    txts = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]
    text = "\n\n".join(txts).strip()
    paragraph_count = len(ps)
    mode = "fallback" if text else "empty"
    meta = {
        "paragraphs": paragraph_count,
        "title": title,
        "domain": domain,
        "mode": mode,
    }
    return text, meta


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


def build_highlighted_html(net_weights, text):
    max_abs = max(abs(v) for v in net_weights.values()) if net_weights else 1
    parts = re.split(r"(\W+)", text)
    spans = []

    for p in parts:
        if p == "":
            continue
        if re.match(r"\W+", p):
            spans.append(p)
            continue
        k = p.lower()
        w = net_weights.get(k, 0)
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
            margin-bottom: 1.2rem;
            letter-spacing: 0.3px;
          }}
          .legend {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.2rem;
            font-size: 0.9rem;
            color: #cccccc;
            flex-wrap: wrap;
          }}
          .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
          }}
          .color-box {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            display: inline-block;
          }}
          .color-box.real {{
            background: rgba(70, 170, 70, 0.85);
          }}
          .color-box.fake {{
            background: rgba(200, 80, 80, 0.85);
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
          <div class="legend">
            <div class="legend-item">
              <span class="color-box real"></span>
              <span>Supports REAL (true) prediction</span>
            </div>
            <div class="legend-item">
              <span class="color-box fake"></span>
              <span>Supports FAKE prediction</span>
            </div>
          </div>
          <div class="article">{rendered}</div>
        </div>
      </body>
    </html>
    """
    return html


def explain(model, tokenizer, text, top_k, threshold):
    predict_fn = predict_factory(model, tokenizer)
    p = predict_fn([text])[0]
    label_idx = int(np.argmax(p))
    exp = explainer.explain_instance(
        text, predict_fn, num_features=top_k, labels=[0, 1]
    )

    try:
        fake_items = exp.as_list(label=0)
    except Exception:
        fake_items = []
    try:
        real_items = exp.as_list(label=1)
    except Exception:
        real_items = []

    fake_dict = {str(k).lower(): v for k, v in fake_items}
    real_dict = {str(k).lower(): v for k, v in real_items}

    tokens = set(list(fake_dict.keys()) + list(real_dict.keys()))
    net_weights_all = {}
    features = []
    for t in tokens:
        sf = fake_dict.get(t, 0.0)
        sr = real_dict.get(t, 0.0)
        net = sr - sf
        if net > 0:
            direction = "real"
        elif net < 0:
            direction = "fake"
        else:
            direction = "neutral"
        net_weights_all[t] = net
        if abs(net) >= threshold:
            features.append(
                {
                    "token": t,
                    "fake_weight": sf,
                    "real_weight": sr,
                    "net_weight": net,
                    "direction": direction,
                }
            )

    if threshold > 0:
        net_weights = {
            t: w for t, w in net_weights_all.items() if abs(w) >= threshold
        }
    else:
        net_weights = net_weights_all

    features.sort(key=lambda x: abs(x["net_weight"]), reverse=True)
    label = "real" if label_idx == 1 else "fake"
    html = build_highlighted_html(net_weights, text)
    return label, p.tolist(), html, features


def text_stats(text: str):
    words = re.findall(r"\w+", text)
    word_count = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)
    return word_count, sentence_count, paragraph_count


st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detection: Model Interpretation")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Model version", ["1.0", "1.1", "1.2", "2.0", "3.0"]
    )
    input_mode = st.radio("Input source", ["URL", "Paste text"])
    top_k = st.slider("Highlighted tokens (top-k)", 5, 50, 20, 5)
    threshold = st.slider(
        "Contribution threshold (|net weight|)", 0.0, 1.0, 0.0, 0.01
    )

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
        extraction_meta = None
        if input_mode == "URL":
            if url:
                txt, extraction_meta = fetch_article_text(url)
                if not txt:
                    st.error("Unable to extract text.")
            else:
                st.warning("Enter a URL.")
        else:
            if txt_manual and txt_manual.strip():
                txt = txt_manual.strip()
                paragraphs = [p for p in txt.split("\n\n") if p.strip()]
                extraction_meta = {
                    "paragraphs": len(paragraphs),
                    "title": "",
                    "domain": "",
                    "mode": "manual",
                }

        if txt:
            txt = txt.lstrip()
            word_count, sentence_count, paragraph_count = text_stats(txt)
            label, probs, html, features = explain(
                model, tokenizer, txt, top_k, threshold
            )
            df_features = pd.DataFrame(features)

            mode_label_map = {
                "structured": "Structured article body",
                "fallback": "Fallback paragraph scan",
                "manual": "Manual text input",
                "empty": "No content extracted",
                "none": "Not available",
            }
            extraction_mode = ""
            extraction_paragraphs = paragraph_count
            article_title = ""
            article_domain = ""
            if extraction_meta:
                extraction_mode = mode_label_map.get(
                    extraction_meta.get("mode", ""),
                    extraction_meta.get("mode", ""),
                )
                extraction_paragraphs = extraction_meta.get(
                    "paragraphs", paragraph_count
                )
                article_title = extraction_meta.get("title", "")
                article_domain = extraction_meta.get("domain", "")

            fake_prob = float(probs[0])
            real_prob = float(probs[1])

            tab_output, tab_highlights, tab_text, tab_features = st.tabs(
                [
                    "Model output",
                    "Explanation highlights",
                    "Extracted article text",
                    "Feature-importance table",
                ]
            )

            with tab_output:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Prediction")
                    st.markdown(
                        f"""
                        <div style="
                            background:#1c1c1c;
                            padding:0.9rem 1.1rem;
                            border-radius:8px;
                            border:1px solid #333;
                        ">
                        <div style="font-size:0.85rem;color:#aaa;">Model version</div>
                        <div style="font-weight:600;">v{model_choice}</div>
                        <div style="margin-top:0.8rem;font-size:0.85rem;color:#aaa;">Classification</div>
                        <div style="font-size:1.2rem;font-weight:700;text-transform:uppercase;">
                        {label}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.subheader("Class probabilities")
                    st.metric("Fake", f"{fake_prob*100:.1f}%")
                    st.metric("Real", f"{real_prob*100:.1f}%")
                with col3:
                    st.subheader("Article summary")
                    st.markdown(
                        f"""
                        <div style="
                            background:#1c1c1c;
                            padding:0.9rem 1.1rem;
                            border-radius:8px;
                            border:1px solid #333;
                            font-size:0.9rem;
                        ">
                        <div><strong>Words:</strong> {word_count}</div>
                        <div><strong>Sentences:</strong> {sentence_count}</div>
                        <div><strong>Paragraphs (text):</strong> {paragraph_count}</div>
                        <div><strong>Paragraphs (extraction):</strong> {extraction_paragraphs}</div>
                        <div style="margin-top:0.4rem;"><strong>Extraction mode:</strong> {extraction_mode or "Unknown"}</div>
                        <div><strong>Domain:</strong> {article_domain or "N/A"}</div>
                        <div><strong>Title:</strong> {article_title or "N/A"}</div>
                        <div style="margin-top:0.4rem;"><strong>Threshold:</strong> |net weight| ≥ {threshold:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with tab_highlights:
                st.subheader("Highlighted explanation")
                components.html(html, height=900, scrolling=True)
                col_html, col_pdf = st.columns(2)
                with col_html:
                    st.download_button(
                        "Download explanation (HTML)",
                        data=html.encode("utf-8"),
                        file_name="explanation.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                pdf_bytes = None
                try:
                    from xhtml2pdf import pisa

                    pdf_io = io.BytesIO()
                    pisa.CreatePDF(io.StringIO(html), dest=pdf_io)
                    pdf_bytes = pdf_io.getvalue()
                except Exception:
                    pdf_bytes = None
                if pdf_bytes:
                    with col_pdf:
                        st.download_button(
                            "Download explanation (PDF)",
                            data=pdf_bytes,
                            file_name="explanation.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

            with tab_text:
                st.subheader("Extracted article text")
                st.text_area("Raw text", txt, height=400)
                st.markdown(
                    f"**Words:** {word_count}  |  **Sentences:** {sentence_count}  |  **Paragraphs:** {paragraph_count}"
                )

            with tab_features:
                st.subheader("Token-level feature importance")
                if not df_features.empty:
                    st.dataframe(
                        df_features[
                            [
                                "token",
                                "direction",
                                "fake_weight",
                                "real_weight",
                                "net_weight",
                            ]
                        ],
                        use_container_width=True,
                    )
                else:
                    st.write(
                        "No feature-importance data available for this instance."
                    )

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
