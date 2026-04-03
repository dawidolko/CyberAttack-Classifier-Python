"""
Download the Cyber Security Attacks dataset from Kaggle into data/cybersecurity_attacks.csv.

Uses Kaggle Hub with cache under .kaggle_cache/ in the project root (no ~/.cache dependency).
Public dataset: https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks
"""

import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEST_CSV = os.path.join(DATA_DIR, "cybersecurity_attacks.csv")


def ensure_dataset(dest_path: str = DEST_CSV, force: bool = False) -> str:
    """
    Ensure cybersecurity_attacks.csv exists. Download from Kaggle if missing or empty.

    Returns path to the CSV file.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if not force and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024:
        return dest_path

    os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(BASE_DIR, ".kaggle_cache"))

    try:
        import kagglehub
    except ImportError as e:
        print("Install dependencies: pip install kagglehub", file=sys.stderr)
        raise e

    print("Downloading dataset from Kaggle (teamincribo/cyber-security-attacks)...")
    folder = kagglehub.dataset_download("teamincribo/cyber-security-attacks")

    csv_src = None
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_src = os.path.join(root, name)
                break
        if csv_src:
            break

    if not csv_src:
        raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")

    shutil.copy2(csv_src, dest_path)
    print(f"Saved: {dest_path} ({os.path.getsize(dest_path) / 1024 / 1024:.2f} MB)")
    return dest_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="Download Cyber Security Attacks CSV from Kaggle.")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = p.parse_args()
    ensure_dataset(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
