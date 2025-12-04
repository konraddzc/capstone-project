# Fake News Detection: ETL Pipeline + Streamlit App

This project implements a complete **Extract–Transform–Load (ETL) pipeline**, **multiple model-training workflows**, and a **Streamlit web interface** for detecting fake vs real news from multiple datasets.

The system:

- Downloads raw datasets from Kaggle
- Cleans, standardizes, and merges text data
- Generates training-ready CSVs and tokenizers
- Trains three models:
  - Mixed Dataset model
  - Iterative multi-dataset model
  - Merged model
- Launches a Streamlit app for real-time inference and model analysis

---

## Project Structure

```
capstone-project/
│
├── scripts/
│   ├── run_app.py
│   ├── run_etl.py
│
├── src/
│   ├── extract/
│   ├── transform/
│   ├── train/
│   ├── utils/
│
├── streamlit/
│   ├── app.py
│   ├── pages/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
├── config/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the project
```
git clone https://github.com/konraddzc/capstone-project
cd capstone-project
```

### 2. Create and activate virtual environment
```
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up Kaggle credentials
Create:

```
%USERPROFILE%/.kaggle/kaggle.json
```

with:

```json
{
  "username": "your_username",
  "key": "your_kaggle_api_key"
}
```

---

## Running the ETL Pipeline

```
run_app dev
```

---

## Training Models

```
python src/train/train_merged_model.py
python src/train/train_iterative_model.py
python src/train/train_individual_models.py (WIP)
```

---

## Streamlit (manual launch)

```
streamlit run streamlit/app.py
```

---

## Tokenizer

- `tokenizer_merged.pkl`

---

## Logging

```
etl_pipeline.log
```

---

## License

MIT License.
