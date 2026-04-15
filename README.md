# Decoding Justice: ML Analysis of Old Bailey Criminal Proceedings (1902–1913)

**LL5532X — Law, Algorithms, and Artificial Intelligence**  
**National University of Singapore**

**Group 3:** Parab Nitin · Pakhale Kalyani Vijay · Shao Lujie · Zhang Xiaoyue  
**Professor:** Ilya Akdemir · April 2026

---

## Overview

This project applies machine learning and natural language processing to **9,192 digitised criminal trial records** from London's Old Bailey (1902–1913) to investigate whether historical legal proceedings encode systematic patterns of prediction, bias, and sentencing.

We address three interconnected research questions:

1. **Verdict Prediction (RQ1):** Can case narrative text predict guilty vs. not guilty verdicts?
2. **Gender Bias in Sentencing (RQ2):** Does defendant gender affect punishment severity after controlling for case-level factors?
3. **Punishment Type Prediction (RQ3):** Can we predict punishment type for convicted defendants?

## Key Findings

- **RQ1:** Verdict prediction achieves **99.33% F1-score** (XGBoost + TF-IDF), though indirect data leakage through sentencing language in trial narratives was identified — itself evidence of how deeply outcomes permeate historical legal text.
- **RQ2:** Female defendants received less harsh punishment (OR ≈ 0.87), consistent with Victorian paternalism, **except** in murder cases where women were punished more harshly (OR > 1.12) — a "double deviance" effect.
- **RQ3:** Punishment type prediction achieves **macro F1 = 0.854** (LightGBM), with text features dramatically improving over structured-only baselines.

Together, the three RQs demonstrate a **bias propagation pipeline**: biased judicial decisions are recorded in trial text → the text encodes those biases → models trained on the text learn to reproduce them.

## Repository Structure

```
├── EDA/
│   ├── 0_xml_extraction.ipynb          # XML data parsing from Old Bailey API
│   ├── 1_EDA Notebook.ipynb            # Descriptive statistics & visualisation
│   ├── 2_Research_based_EDA.ipynb      # NLP feature engineering & close reading
│   └── EDA CSV Outputs/               # Intermediate EDA outputs
│
├── Modeling/
│   ├── Data Preparation for Modeling/  # Cleaning, anomaly resolution, feature prep
│   ├── Research Question 1/           # Verdict prediction (Logistic Regression, SVM, XGBoost)
│   ├── Research Question 2/           # Gender bias analysis (progressive logistic regression)
│   └── Research Question 3/           # Punishment type prediction (LightGBM)
│
├── Proposal & Final Report/
│   ├── Project Proposal.pdf           # Initial project proposal
│   ├── Group3_Research_Report_Final.docx
│   └── Group3_Research_Report_Final.md
│
├── Slide/                             # Presentation decks
├── Notebooks in HTML/                 # HTML exports of all notebooks
├── report_charts/                     # All figures used in the report
│
├── Old Bailey Proceedings data_1900-1913.csv   # Raw dataset
└── README.md
```

## Data

The dataset originates from the [Old Bailey Proceedings Online](https://www.oldbaileyonline.org/), parsed from XML via the DHI Old Bailey API.

| Property            | Value                              |
|---------------------|------------------------------------|
| Total trial records | 9,192                              |
| Period              | 1902–1913                          |
| Raw features        | 19                                 |
| Engineered features | 58                                 |
| Verdict split       | ~80% guilty / ~20% not guilty      |
| Gender split        | ~90% male / ~10% female            |
| Dominant crime type | Theft (>50% of all trials)         |

## Methodology

The project follows the **DRME pipeline** (Data → Representation → Model → Evaluation):

- **EDA & Feature Engineering:** Text cleaning, tokenisation, lemmatisation (spaCy), POS tagging, named entity extraction, VADER sentiment, LDA topic modelling, readability indices. Vocabulary reduced from 62,488 to 34,210 tokens.
- **RQ1 — Verdict Prediction:** Binary classification using TF-IDF + Logistic Regression, Naive Bayes, SVM, and XGBoost. Includes data leakage analysis.
- **RQ2 — Gender Bias:** Progressive logistic regression isolating the gender effect while controlling for crime type, text length, and temporal factors.
- **RQ3 — Punishment Type:** Multiclass classification (4 punishment groups) using LightGBM with structured + text features.

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/zhangxiaoyue311-cmd/LL5532X.git
   cd LL5532X
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm spacy matplotlib seaborn nltk
   python -m spacy download en_core_web_sm
   ```

3. **Run notebooks in order**
   - Start with `EDA/0_xml_extraction.ipynb` for data extraction
   - Proceed through `EDA/1_EDA Notebook.ipynb` and `2_Research_based_EDA.ipynb`
   - Run `Modeling/Data Preparation for Modeling/` before the RQ notebooks
   - Execute each Research Question notebook independently

## License

This project is for academic purposes as part of the LL5532X module at NUS.
