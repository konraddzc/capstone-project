import sys
import subprocess
from pathlib import Path
import argparse

from scripts.run_etl import main as run_etl_main
from config.env_config import setup_env, ENVS  # ENVS = ['dev', 'test', 'prod']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=ENVS, help="Environment name")
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip running the ETL step",
    )
    args = parser.parse_args()

    setup_env(["run_app", args.env])

    if not args.skip_etl:
        run_etl_main()

    base_dir = Path(__file__).resolve().parent.parent
    streamlit_entry = base_dir / "streamlit" / "app.py"

    subprocess.run(["streamlit", "run", str(streamlit_entry)])


if __name__ == "__main__":
    main()
