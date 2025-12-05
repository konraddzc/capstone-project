from pathlib import Path
import os
import shutil
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_CACHE = PROJECT_ROOT / "data" / "kaggle_cache"
KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(KAGGLE_CACHE)


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return matches[0]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def extract_fake_true():
    path = Path(
        kagglehub.dataset_download(
            "clmentbisaillon/fake-and-real-news-dataset"
        )
    )
    fake_src = find_file(path, "Fake.csv")
    true_src = find_file(path, "True.csv")
    copy_file(fake_src, RAW_DIR / "Fake.csv")
    copy_file(true_src, RAW_DIR / "True.csv")


def extract_welfake():
    path = Path(
        kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    )
    welfake_src = find_file(path, "WELFake_Dataset.csv")
    copy_file(welfake_src, RAW_DIR / "WELFake_Dataset.csv")


def extract_ainews():
    path = Path(
        kagglehub.dataset_download("asminshan/ai-fake-news-prediction")
    )
    ainews_src = find_file(path, "AI_news.csv")
    copy_file(ainews_src, RAW_DIR / "AI_news.csv")


def main():
    extract_fake_true()
    extract_welfake()
    extract_ainews()


if __name__ == "__main__":
    main()
