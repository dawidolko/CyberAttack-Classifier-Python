# Cyber Security Attacks — Multiclass Classification

End-to-end machine learning project for **multiclass classification** of network attacks using the [Cyber Security Attacks](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks) dataset from Kaggle. The primary model is a **Random Forest** classifier; **Gradient Boosting** and **k-NN** are trained for comparison. A **Streamlit** dashboard visualizes exploratory analysis, preprocessing, training, and evaluation.

## Features

- **40,000** instances, **25** attributes (see the dashboard *Project Overview* for descriptions).
- **Target:** `Attack Type` with three nearly balanced classes: **DDoS**, **Malware**, **Intrusion**.
- **Pipeline** (`pipeline.py`): EDA plots, preprocessing (drops non-generalizing columns, binary flags for sparse fields, label encoding, scaling), 80/20 stratified split, Random Forest + comparison models, 5-fold CV, test metrics, confusion matrix, ROC (OvR), feature importance. Artifacts go to `results/`; the trained RF is saved under `models/random_forest.joblib`.
- **Dashboard** (`app.py`): Tabbed Streamlit UI (overview, EDA, preprocessing, model design, **model comparison**, detailed RF results, interactive explorer).
- **Data download** (`download_data.py`): Fetches the official CSV via [Kaggle Hub](https://github.com/Kaggle/kagglehub) into `data/cybersecurity_attacks.csv` (cache directory: `.kaggle_cache/`).

## Requirements

- **Python 3.10–3.13** (tested with 3.12).
- Internet access on **first run** to download the dataset (~5 MB) unless `data/cybersecurity_attacks.csv` is already present.

## Quick start (one command)

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows

Double-click `start.bat` or run in `cmd` / PowerShell:

```bat
start.bat
```

The script will:

1. Resolve Python 3.10+.
2. Create `venv/` if needed and `pip install -r requirements.txt`.
3. Run `python pipeline.py` (downloads data if missing, trains models, writes `results/`).
4. Start the Streamlit app at **http://localhost:8501** (`streamlit run app.py`).

Stop the server with **Ctrl+C**.

### Manual steps (optional)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_data.py           # optional; pipeline also downloads if needed
python pipeline.py
streamlit run app.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `download_data.py` | Download dataset from Kaggle into `data/` |
| `pipeline.py` | Full ML pipeline and evaluation |
| `app.py` | Streamlit dashboard |
| `data/` | Dataset CSV (generated; see `.gitignore`) |
| `results/` | Metrics, JSON, PNG plots, NumPy confusion matrix (generated) |
| `models/` | Saved `random_forest.joblib` (generated) |
| `start.sh` / `start.bat` | One-command setup + pipeline + Streamlit |

## Academic report (Polish course outline)

For the *Sztuczna inteligencja* report, map sections as follows:

1. **Student data** — name, program, year, academic year (fill in manually).
2. **Course** — Artificial Intelligence (or your exact course title).
3. **Project topic** — Multiclass classification of cyber security attacks (Random Forest on Kaggle dataset).
4. **Problem characterization** — Supervised multiclass classification; balanced three-class target; network and security features with missing values in several columns.
5. **Number of instances** — 40,000.
6. **Attributes** — 25; use the table in the Streamlit *Project Overview* tab and the dataset documentation on Kaggle.
7. **Preprocessing** — Summarize steps from the *Preprocessing* tab / `preprocessing_info.json` (dropped columns, binary flags, encodings, scaling, split).
8. **Model design** — Random Forest (primary), plus Gradient Boosting and k-NN for comparison; hyperparameters as in the *Model & Training* tab and `pipeline.py`.
9. **Results** — Accuracy, macro F1, precision, recall, ROC AUC, confusion matrix, per-class metrics (`metrics.json` / dashboard).
10. **Conclusions** — Strengths of RF on this task, role of important features, limitations (e.g. label encoding of IPs removed; text fields dropped).

## Notes

- **Empirical performance:** On this Kaggle release, **test accuracy is often near the random baseline (≈1/3)** for balanced three-class prediction, and **ROC AUC is near 0.5**, with all three models behaving similarly. That is a valid **finding** for your report: after removing identifiers and free text, the remaining tabular features may carry **little usable signal** for `Attack Type` in this synthetic split. The pipeline and metrics are still correct; interpret results honestly in section *Wnioski* / *Conclusions*.
- **Git:** `venv/`, `.kaggle_cache/`, `data/cybersecurity_attacks.csv`, and generated `results/` / `models/` artifacts are listed in `.gitignore`. Clone the repo and run `./start.sh` to regenerate everything.
- **Kaggle authentication:** Public dataset download via `kagglehub` typically works without extra setup; if you hit auth errors, follow [Kaggle API credentials](https://www.kaggle.com/docs/api) and set `KAGGLE_USERNAME` / `KAGGLE_KEY` or place `kaggle.json` in `~/.kaggle/`.

## License

Dataset usage is subject to the [Kaggle dataset license](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks). This repository code is provided for educational use.
