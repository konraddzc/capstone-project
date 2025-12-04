import sys
import subprocess
from pathlib import Path

from scripts.run_etl import main as run_etl_main
from config.env_config import setup_env


def main():
    setup_env(sys.argv)

    run_etl_main()

    base_dir = Path(__file__).resolve().parent.parent
    streamlit_entry = base_dir / "streamlit" / "app.py"

    subprocess.run(["streamlit", "run", str(streamlit_entry)])


if __name__ == "__main__":
    main()
