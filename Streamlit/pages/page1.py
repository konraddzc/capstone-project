from pathlib import Path
import datetime

import pandas as pd
import streamlit as st
import graphviz


st.set_page_config(
    page_title="Fake News Detection Dashboard",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "notebooks"
LOG_FILE = PROJECT_ROOT / "etl_pipeline.log"

st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.20rem;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0.7rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #d4d4d8;
        color: #4b5563;
        background: rgba(249, 250, 251, 0.7);
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .badge-dark-ok {
        border-color: #16a34a;
        color: #16a34a;
        background: rgba(22, 163, 74, 0.08);
    }
    .badge-dark-miss {
        border-color: #dc2626;
        color: #dc2626;
        background: rgba(220, 38, 38, 0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
    }
    .metric-muted {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    hr {
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.35);
        margin: 1.4rem 0 1.1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_dataset_stats():
    files = {
        "Fake or Real Dataset": DATA_PROCESSED / "fake_or_real_news_clean.csv",
        "WELFake Dataset": DATA_PROCESSED / "welfake_clean.csv",
        "AI news Dataset": DATA_PROCESSED / "ainews_clean.csv",
        "Merged Dataset": DATA_PROCESSED / "news_merged_clean.csv",
    }
    stats = {}
    for name, path in files.items():
        if path.exists():
            try:
                n_rows = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
                stats[name] = n_rows if n_rows >= 0 else None
            except Exception:
                stats[name] = None
        else:
            stats[name] = None
    return stats


def get_model_stats():
    files = {
        "v1.0 baseline": MODELS_DIR / "best_model.keras",
        "v1.1 finetuned": MODELS_DIR / "finetuned_model.keras",
        "v1.2 gen3model": MODELS_DIR / "v3_model.keras",
        "v2.0 curriculum": MODELS_DIR / "curriculum_best_model.keras",
        "v3.0 merged": MODELS_DIR / "merged_model.keras",
    }
    stats = {}
    for name, path in files.items():
        stats[name] = path.exists()
    return stats


def get_last_etl_run():
    if LOG_FILE.exists():
        ts = datetime.datetime.fromtimestamp(LOG_FILE.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return None


st.title("Fake News Detection Dashboard")

st.markdown(
    """
This dashboard provides an overview of the fake news detection system, including the data pipeline, 
trained models, and their current status.

Use the sidebar to navigate to:
- **Fake News Detection Tool** – analyse single articles and view model explanations.
- **Model Analysis** – compare model performance quantitatively across datasets.
"""
)

st.markdown("<hr>", unsafe_allow_html=True)

ds_stats = get_dataset_stats()
model_stats = get_model_stats()
last_etl = get_last_etl_run()

top_left, top_mid, top_right = st.columns([1.2, 1.1, 1.1])

with top_left:
    st.markdown(
        """
        <div class="section-title">System status</div>
        <div class="section-subtitle">
            High-level indicators of data availability and model readiness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        ready_datasets = sum(1 for v in ds_stats.values() if v is not None)
        st.markdown(
            "<div class='metric-label'>Datasets ready</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric-value'>{ready_datasets} / {len(ds_stats)}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        ready_models = sum(1 for v in model_stats.values() if v)
        st.markdown(
            "<div class='metric-label'>Models available</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric-value'>{ready_models} / {len(model_stats)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(
        "<div class='metric-label'>Last ETL pipeline run</div>",
        unsafe_allow_html=True,
    )
    if last_etl:
        st.markdown(
            f"<div class='metric-value'>{last_etl}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='metric-muted'>No ETL log detected</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with top_mid:
    st.markdown(
        """
        <div class="section-title">Datasets overview</div>
        <div class="section-subtitle">
            Cleaned datasets available for training and evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for name, n_rows in ds_stats.items():
        if n_rows is None:
            st.markdown(f"- **{name}**: not available")
        else:
            st.markdown(f"- **{name}**: {n_rows:,} samples")

    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown(
        """
        <div class="section-title">Models overview</div>
        <div class="section-subtitle">
            Deployed model versions and their roles in the pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for name, ok in model_stats.items():
        cls = "badge badge-dark-ok" if ok else "badge badge-dark-miss"
        status = "ready" if ok else "missing"
        st.markdown(
            f"<span class='{cls}'>{name}: {status}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

mid_left, mid_right = st.columns([1.1, 1])

with mid_left:
    st.markdown(
        """
        <div class="section-title">End-to-end pipeline</div>
        <div class="section-subtitle">
            Schematic view of the ETL and model lifecycle.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#d4d4d8",
        fillcolor="#f4f4f5",
        fontname="Helvetica",
        fontsize="10",
    )
    dot.attr("edge", color="#9ca3af")

    dot.node("src", "Raw news sources\n(Kaggle datasets)")
    dot.node("ext", "Extract\n(KaggleHub → data/raw)")
    dot.node("tr", "Transform\n(cleaning, labels,\nprocessed CSVs)")
    dot.node("tok", "Tokenize\n(vocabulary + sequences)")
    dot.node("train", "Train models\n(v1.0–v3.0)")
    dot.node("eval", "Evaluate\n(metrics, ROC, CM)")
    dot.node("serve", "Serve\n(Streamlit pages)")

    dot.edge("src", "ext")
    dot.edge("ext", "tr")
    dot.edge("tr", "tok")
    dot.edge("tok", "train")
    dot.edge("train", "eval")
    dot.edge("train", "serve")
    dot.edge("eval", "serve")

    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with mid_right:
    st.markdown(
        """
        <div class="section-title">How to use this dashboard</div>
        <div class="section-subtitle">
            Recommended workflow for model inspection and evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
1. Verify that ETL has run and that datasets are present in the **System status** and **Datasets overview** panels.
2. Use the **Fake News Detection Tool** page to analyse single articles and inspect token-level explanations.
3. Use the **Model Analysis** page to compare model versions across datasets using accuracy, F1, ROC and confusion matrices.
4. Iterate on preprocessing or training settings, re-run ETL and training, and monitor the impact here.
"""
    )

    st.markdown("</div>", unsafe_allow_html=True)
