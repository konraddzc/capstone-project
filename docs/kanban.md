# Kanban

---

## 1. ETL Pipeline

### Done
- Load raw datasets (Fake/Real, WELFake, AI News).
- Standardize columns, normalize labels to 0/1.
- Text cleaning: regex normalization, lemmatization, stopword removal.
- Save cleaned outputs to `data/processed/`.
- Log ETL run timestamp (`etl_pipeline.log`).

### Backlog
- Improve cleaning (e.g. boilerplate removal).
---

## 2. Tokenization & Preprocessing

### Done
- Define preprocessing pipeline (tokenize → lemmatize → stopword removal).
- Train tokenizers for:
  - Baseline
  - Iterative curriculum
  - Merged dataset
- Save tokenizers to `notebooks/`.

### Backlog
- Switch to subword tokenization.
- Move tokenizers fully to ETL pipeline
---

## 3. Model Training

### Done
- Train and save versions:
  - v1.0 baseline  
  - v1.1 finetuned  
  - v1.2 v3  
  - v2.0 curriculum  
  - v3.0 merged  
- Save `.keras` models + tokenizer mapping.

### Backlog
- Add training automation fully to pipeline

---

## 4. Streamlit – Main Dashboard

### Done
- Status indicators: datasets available, models available, last ETL.
- Dataset + model overview panels.
- Pipeline diagram.
- Usage instructions.

### Backlog
- Add environment + dependency checks

---

## 5. Streamlit – Fake News Detector + LIME

### Done
- Support URL extraction and manual text input.
- LIME explanations with fake/real contributions.
- Highlighted HTML with token coloring.
- Tabs for:
  - Model output
  - Highlights
  - Raw article text
  - Feature-importance table

### Backlog
- Add sentence-level explanations.
- Add model comparison

---

## 6. Streamlit – Model Analysis

### Done
- Compare two model versions on selected dataset.
- Random sampling with seed.
- Metrics: accuracy, precision/recall/F1, confusion matrix, ROC.
- Classification reports in tabs.

### Backlog
- Export full comparison report.
- Explanation of each metric and importance
- Deeper dive into comparison between models

---

## 7. Maintenance

### Done
- Caching for models, tokenizers, datasets.
- Basic runtime error handling.
- Unit tests

### Backlog
- component tests
- e2e tests
- integration tests
- own code comments