# CaixaBank Fraud Detection -- Hackathon 2026

End-to-end pipeline for a banking fraud-detection system: cleaning,
merging, feature engineering, LightGBM training, and three Streamlit
apps (real-time fraud simulator, BI dashboard, security command
center).



# 🛡️ FinShield AI: End-to-End Banking Fraud & Financial Analytics

## 📌 Project Overview
**FinShield AI** is an enterprise-grade FinTech ecosystem built on the CaixaBank Hackathon dataset. This project demonstrates a complete end-to-end Machine Learning and Data Engineering pipeline, starting from raw transactional data up to deploying interactive real-time executive dashboards.

The primary challenge tackled in this project was detecting highly imbalanced fraudulent transactions (a `~0.15%` fraud rate) while simultaneously extracting deep financial insights into credit risk and customer spending behaviors.

---

## 🛠️ Tech Stack
* **Data Engineering & Analysis:** Pandas, NumPy, PyArrow, FastParquet
* **Machine Learning:** LightGBM, Scikit-Learn
* **Dashboards & UI:** Streamlit, Plotly Express
* **Version Control & Deployment:** Git, GitHub, Streamlit Community Cloud

---

## 🚀 The Engineering Journey (Step-by-Step)

### Step 1: Data Integration & ETL
* **Challenge:** The raw data was scattered across massive CSV and JSON files containing millions of transaction records, user profiles, and merchant details.
* **Action:** Built a robust ETL pipeline to merge these relational datasets securely. Cleaned missing values and downcasted data types to optimize memory usage.

### Step 2: Advanced Feature Engineering
* **Challenge:** Raw data alone isn't enough to catch smart fraudsters or assess credit risk.
* **Action:** Engineered powerful behavioral and financial indicators. 
  * Created `tx_to_limit_ratio` (Transaction Amount / Credit Limit) to monitor credit utilization.
  * Created `debt_to_income_ratio` to flag high-risk customers.
  * Extracted temporal features (hour, day of week) to detect unusual banking hours.

### Step 3: Model Training & Handling Extreme Imbalance
* **Challenge:** The fraud rate was microscopic (`~0.15%`), meaning a naive model could achieve 99.85% accuracy by simply predicting "Not Fraud" every time.
* **Action:** * Selected **LightGBM** for its blazing speed on millions of rows.
  * Applied `scale_pos_weight` to heavily penalize the model for missing a fraudulent transaction.
  * Ignored "Accuracy" and evaluated the model strictly using **AUPRC (Area Under the Precision-Recall Curve)** and **Recall**, achieving an impressive **83.5% Recall** (successfully catching 83.5% of all frauds).

### Step 4: Eradicating Data Leakage
* **Challenge:** The model initially showed abnormally high importance for features like `client_id` and `merchant_id`. It was *memorizing* victims rather than learning fraud *patterns*.
* **Action:** Explicitly removed all unique identifiers before final training to ensure the model generalizes perfectly to completely new customers and merchants.

### Step 5: Bypassing Deployment Constraints (GitHub 100MB Limit)
* **Challenge:** The final merged dataset (`train_features.parquet`) was ~575MB, making it impossible to push to GitHub for deployment.
* **Action:** Implemented **Stratified Sampling** via Scikit-Learn to extract a representative subset of 250,000 rows (`train_features_sample.parquet` - ~15MB). This perfectly maintained the 0.15% fraud distribution while keeping the dashboards lightning-fast on the web.

### Step 6: Dual-Engine Dashboards Deployment
* **Action:** Built two specialized interactive apps using Streamlit:
  1. **Security Shield (`app_security.py`):** A real-time Fraud Simulator. Analysts can input transaction details, and the LightGBM model instantly outputs a risk probability and an Approved/Declined alert. Fixed categorical feature dictionary mismatch errors during single-row inference.
  2. **Portfolio Health (`app_financial.py`):** An executive BI dashboard utilizing Plotly to visualize credit risk thresholds, spending by age, and top merchant categories.

---

## 💻 How to Run the Project Locally

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/YourUsername/FinShield-AI-Banking-System.git](https://github.com/YourUsername/FinShield-AI-Banking-System.git)
cd FinShield-AI-Banking-System
pip install -r requirements.txt

## Project layout

```
CaixaBank/
|-- apps/                        # Streamlit applications
|   |-- app.py                   # Real-time fraud simulator
|   |-- app_financial.py         # BI portfolio dashboard
|   |-- app_security.py          # Security command center
|
|-- data/
|   |-- raw/                     # Original CSV / JSON exports
|   |   |-- transactions_data.csv
|   |   |-- cards_data.csv
|   |   |-- users_data.csv
|   |   |-- mcc_codes.json
|   |   |-- train_fraud_labels.json
|   |
|   |-- processed/               # Pipeline outputs (parquet)
|   |   |-- cleaned_transactions.parquet
|   |   |-- cleaned_cards.parquet
|   |   |-- cleaned_users.parquet
|   |   |-- master_banking_data.parquet
|   |   |-- train_features.parquet
|   |   |-- inference_features.parquet
|   |
|   |-- sample/                  # GitHub-friendly stratified sample
|       |-- train_features_sample.parquet  (~16 MB)
|
|-- models/                      # Trained artefacts
|   |-- lgbm_fraud_model.txt
|
|-- src/                         # Pipeline scripts
|   |-- clean_banking_tables.py
|   |-- build_master_abt.py
|   |-- engineer_banking_features.py
|   |-- train_banking_fraud_model.py
|   |-- create_sample.py
|
|-- docs/                        # Diagrams / write-ups
```

## Pipeline order

1. `python src/clean_banking_tables.py` -- clean the raw CSVs into
   three parquet files in `data/processed/`.
2. `python src/build_master_abt.py` -- merge the three parquet files
   plus the two JSON enrichment files into a single master ABT.
3. `python src/engineer_banking_features.py` -- engineer the financial
   and temporal features and split into train / inference parquets.
4. `python src/train_banking_fraud_model.py` -- fit a LightGBM
   classifier and write it to `models/lgbm_fraud_model.txt`.
5. `python src/create_sample.py` -- produce a stratified 250K-row
   sample for GitHub-friendly deployment.

## Apps

```
streamlit run apps/app.py
streamlit run apps/app_financial.py
streamlit run apps/app_security.py
```

All three apps resolve their inputs via `PROJECT_ROOT` defined at the
top of the file, so they work no matter what directory you invoke
`streamlit run` from.

## Notes

* `data/processed/train_features.parquet` is intentionally **not**
  committed to GitHub (it is ~480 MB). The same goes for
  `master_banking_data.parquet` (~575 MB) and
  `cleaned_transactions.parquet` (~265 MB). Only the
  `data/sample/train_features_sample.parquet` (~16 MB) and
  `models/lgbm_fraud_model.txt` (~225 KB) are deployable artefacts.
* The fraud rate in every file is preserved at ~0.15% via stratified
  sampling.
