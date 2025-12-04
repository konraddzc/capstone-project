import streamlit as st

st.set_page_config(
    page_title="Fake News Detection Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .section-card {
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #333333;
        background: #111111;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #bbbbbb;
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #555;
        color: #cccccc;
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
    }
    hr {
        border: none;
        border-top: 1px solid #333333;
        margin: 2rem 0 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fake News Detection Dashboard")

st.markdown(
    """
Welcome to the fake news detection project dashboard.

Use the navigation (in the sidebar) to:
- Inspect how the model classifies **individual articles**.
- Evaluate and compare model performance on **full datasets**.
"""
)

st.markdown("---")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">1. Fake News Detection: Model Interpretation</div>
        <div class="section-subtitle">
            Analyse a single news article and see how the model arrives at its prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**What you can do on that page**

- Choose a **model version** (e.g. 1.0, 1.1, 2.0, 3.0).
- Provide an article via:
  - **URL** (automatic article text extraction), or  
  - **Paste text** directly.
- Run the detector to get:
  - A **fake / real** classification with class probabilities.
  - A **token-level explanation** (highlighted text) showing which words
    pushed the prediction towards *fake* or *real*.

**Suggested workflow**

1. Start with model `3.0` (merged model) to get the latest behaviour.
2. Paste or load an article.
3. Inspect which phrases are highlighted as evidence for each class.
4. Switch model versions to see how explanations and predictions differ.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">2. Model Analysis</div>
        <div class="section-subtitle">
            Evaluate models on full datasets and compare their overall performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**What you can do on that page**

- Select a **model version** (1.0 / 1.1 / 2.0 / 3.0).
- Pick a **dataset**.
- Choose an **evaluation subset size** and a **random seed**.
- Run evaluation to obtain:
  - Accuracy, precision, recall, F1 score.
  - **Confusion matrix** (fake vs real).
  - **ROC curve** and AUC.
  - A detailed **classification report** (per-class metrics).

**Suggested workflow**

1. Run all models on the same dataset and subset size.
2. Record accuracy, F1 for the *real* class, and AUC.
3. Use `news_merged.csv` to check how well the merged model (`3.0`)
   generalises across sources.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Models available</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="badge">v1.0</div> baseline best model  
<div class="badge">v1.1</div> finetuned on extra data  
<div class="badge">v2.0</div> curriculum-trained model  
<div class="badge">v3.0</div> merged model (tokenizer + data)
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**Notes**

- All models are binary classifiers: **0 = fake**, **1 = real**.
- Tokens are pre-processed using:
  - lowercasing  
  - lemmatization  
  - English stopword removal  
  - removal of tokens shorter than 4 characters
"""
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">How to navigate</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
- Use the **sidebar** to switch between:
    - *Fake News Detection Tool*
    - *Model Analysis*
- Return here any time to recall what each page does and how to use it.
"""
    )
