# AI Maturity Index

This project develops an interpretable, data-driven **AI Maturity Index** for a consolidated set of global companies.  
It combines **natural language processing (NLP)**, **financial & innovation metrics**, and **machine learning** to estimate each company’s AI maturity level.

The work follows the methodology proposed in the topic:

> **"Assessing AI Maturity with Advanced Methods – Data-driven analysis"**

as part of the MSc *Management and Technology* Master’s Thesis (TUM).

---

## Objective

The goal of this project is to:
- Integrate and clean several corporate datasets (HG Insights, IMD, Fortune, Drucker Institute)
- Develop a unified **Master Company List** with consistent identifiers
- Engineer meaningful AI-related features using **NLP on corporate disclosures**, **innovation**, and **financial indicators**
- Train and evaluate **machine learning models** to predict or approximate AI maturity
- Produce an interpretable **AI Maturity Index (0–100)** with explainable sub-scores

---

## Data Sources

| Source | Description | N |
|--------|--------------|---|
| [HG Insights](https://explore.hginsights.com/ai-1000-2025) | AI-Active Companies 2025 | 1,000 |
| [IMD AI Maturity Index](https://www.imd.org/artificial-intelligence-maturity-index/results/#_tab_List) | Benchmark data, serves as primary label | 200 |
| [Fortune Global 500](https://us500.com/fortune-global-500) | Global corporations | 500 |
| [Fortune 1000 US](https://us500.com/fortune-1000-companies) | US corporations | 1,000 |
| [Drucker Institute](https://drucker.institute/annual-data/annual-ranking-data-2024/) | Annual rankings, includes Innovativeness | 426 |

Additional data sources may include:
- Job postings (LinkedIn / Indeed scraping)
- Patent databases (AI-related CPC codes)
- Google Trends & News analytics
- Corporate disclosures (10-K, ESG reports)
- Web data (company websites, press releases)

---

## Methodology Overview

1. **Data Integration & Cleaning**
   - Merge and deduplicate company lists using fuzzy matching (`rapidfuzz`)
   - Standardize identifiers (names, domains, tickers)

2. **Feature Engineering**
   - **NLP features:** AI-related mentions in corporate texts (via `sentence-transformers`)
   - **Financial metrics:** R&D intensity, CapEx ratio, margins
   - **Innovation proxies:** patents, Drucker Innovativeness score
   - **Talent signals:** AI/ML-related job postings
   - **Market signals:** Google Trends, GitHub activity

3. **Modeling**
   - Gradient boosting (LightGBM/XGBoost/CatBoost)
   - Ordinal or multitask regression (e.g., IMD + Drucker labels)
   - Feature importance & interpretability via SHAP

4. **Evaluation**
   - Metrics: Spearman’s ρ, Kendall’s τ, MAE
   - Cluster analysis & bias assessment (industry/region)

5. **Outputs**
   - AI Maturity Index (0–100)
   - Subscores: *Strategy, Talent, Infrastructure, Innovation, Adoption*
   - Explainability plots (e.g. SHAP, feature contribution)

---

## Repository Structure

ai-maturity-index/
├─ data_raw/        # Original datasets (CSV/XLSX)
├─ data_clean/      # Cleaned & deduplicated data
├─ notebooks/       # Stepwise Jupyter notebooks
│  ├─ 01_ingest_clean.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_modeling.ipynb
│  └─ 04_evaluation.ipynb
├─ src/             # Helper modules and reusable functions
│  ├─ cleaning.py
│  ├─ features.py
│  ├─ modeling.py
│  └─ visualization.py
├─ outputs/         # Model results, charts, and final index
├─ requirements.txt # Python dependencies
└─ README.md        # Project documentation

---

## Setup

### 1. Clone the repository
$ git clone https://github.com/timokoba/ai-maturity-index.git
$ cd ai-maturity-index

### 2. Create a Conda environment
$ conda create -n aimaturity python=3.10
$ conda activate aimaturity

### 3. Install dependencies
$ pip install -r requirements.txt

### 4. Register the Jupyter kernel
$ python -m ipykernel install --user --name aimaturity --display-name "Python (aimaturity)"

---

## Example Workflow
| Step | Notebook                      | Description                                         |
|-----:|-------------------------------|-----------------------------------------------------|
| 1    | 01_ingest_clean.ipynb         | Load & merge datasets, deduplicate company names    |
| 2    | 02_feature_engineering.ipynb  | Build AI-related features (text, finance, talent)   |
| 3    | 03_modeling.ipynb             | Train machine learning models                       |
| 4    | 04_evaluation.ipynb           | Evaluate, explain, and export the AI Maturity Index |

---

## Requirements
Python ≥ 3.10

Main dependencies:
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost,
sentence-transformers, transformers, shap, matplotlib, seaborn,
beautifulsoup4, requests, lxml, statsmodels, polars

---

## License
This project is licensed under the MIT License — feel free to use, adapt, and reference it with attribution.

---

## Author
Timo Koba  
TUM M.Sc., Management and Technology  
LinkedIn: https://www.linkedin.com/in/timo-koba  
GitHub:  https://github.com/timokoba


