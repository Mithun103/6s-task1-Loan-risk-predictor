# LoanRiskPredictor

A machine learning project designed to predict loan default risk with a focus on maximizing recall to minimize financial losses from undetected high-risk loans.

## Project Overview

LoanRiskPredictor utilizes multiple machine learning models, preprocessing pipelines, and feature engineering techniques to provide robust loan default predictions. The system prioritizes **recall over precision** to ensure maximum identification of potential defaults, protecting financial institutions from significant losses.

**Key Achievement**: 70.23% recall rate, successfully identifying 7 out of every 10 loans that would default.

## Dataset Overview

- **Records**: 255,246 loan applications
- **Features**: Numerical and categorical variables
- **Target**: Default (0 = No Default, 1 = Default)
- **Class Distribution**: Highly imbalanced (7.5:1 ratio)
  - Non-defaults: ~225,000 applications
  - Defaults: ~30,000 applications

## Repository Structure

```
LoanRiskPredictor/
│
├── api/
│   ├── __init__.py
│   └── fastapi_app.py          # FastAPI application to serve ML predictions
│
├── data/                       # Raw and processed datasets
│
├── models/                     # Trained model artifacts
│   ├── adaboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── logistic_regression.joblib
│   ├── preprocessing_pipeline.joblib
│   └── voting_model.pkl
│
├── notebooks/                  # Jupyter notebooks
│   └── eda_and_visualization.ipynb
│
├── src/                        # Source code for ML pipeline
│   ├── __init__.py
│   ├── config.py               # Paths and configuration constants
│   ├── inference.py            # ML inference and feature importance
│   └── preprocessing.py        # Data preprocessing scripts
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Model Performance

### Why Recall Matters in Risk Prediction

In loan default prediction, **recall is our primary metric** because:

- **Cost Asymmetry**: Missing a default costs the entire loan amount (~$15,000 average), while rejecting a good applicant only costs potential interest revenue (~$2,000)
- **Risk Management**: Financial institutions must identify high-risk loans to protect portfolio health
- **Regulatory Compliance**: Banks need demonstrable risk assessment capabilities
- **Business Impact**: 70% recall translates to preventing ~70% of potential loan losses

### Model Comparison

| Model | Accuracy | **Recall** | Precision | F1-Score | ROC AUC |
|-------|----------|------------|-----------|----------|---------|
| **Logistic Regression** ⭐ | 68.14% | **70.23%** | 22.31% | 33.86% | 75.63% |
| AdaBoost | 68.91% | **69.10%** | 22.59% | 34.05% | 75.70% |
| LightGBM | 69.74% | **66.89%** | 23.00% | 34.00% | - |

**Selected Model**: Logistic Regression (highest recall + interpretability)

### Detailed Results

#### Logistic Regression (Production Model)
```
Classification Report:
              precision    recall  f1-score   support
           0       0.95      0.68      0.79     45121
           1       0.22      0.70      0.34      5929
    accuracy                           0.68     51050
   macro avg       0.58      0.69      0.56     51050
weighted avg       0.86      0.68      0.74     51050

Confusion Matrix:
[[30622 14499]
 [ 1765  4164]]
```

#### AdaBoost
```
Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.69      0.80     45121
           1       0.23      0.69      0.34      5929
    accuracy                           0.69     51050
   macro avg       0.59      0.69      0.57     51050
weighted avg       0.86      0.69      0.74     51050

Confusion Matrix:
[[31084 14037]
 [ 1832  4097]]
```

## Exploratory Data Analysis (EDA)

### Key Findings

**Numerical Features**:
- Most features show uniform distributions (suggesting synthetic data)
- **InterestRate**: Strongest predictor (defaulters: ~17% vs non-defaulters: ~13%)
- **Age**: Younger applicants slightly more likely to default
- **Income & Employment**: Defaulters have lower income and shorter employment
- Near-zero correlations between features (independence)

**Categorical Features**:
- Perfectly balanced distributions across all categories
- Minimal predictive power individually
- Slight signals from EmploymentType and Education

**Target Distribution**:
- Highly imbalanced: requires careful handling during training
- Class weights and specialized metrics (recall) are essential

### Feature Correlation with Target
- InterestRate: +0.13 (strongest)
- Age: -0.17
- LoanAmount: +0.09
- Income: -0.10
- MonthsEmployed: -0.10

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/username/LoanRiskPredictor.git
cd LoanRiskPredictor
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate        # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the API


### Note
- Ensure that the `models/` directory contains all trained model files.
- `src/config.py` must have correct paths to model and preprocessor artifacts.
- The FastAPI app automatically loads the model and preprocessing pipeline on startup.

**Start FastAPI server**:
```bash
uvicorn LoanRiskPredictor.api.fastapi_app:app --reload
```

**Access documentation**: `http://127.0.0.1:8000/docs`

## API Usage


### Endpoints

#### GET /
Returns welcome message.

#### POST /predict
Predicts loan default risk and returns feature importance.

**Example Request**:
```json
{
  "Age": 35,
  "Income": 75000,
  "LoanAmount": 15000,
  "CreditScore": 680,
  "MonthsEmployed": 60,
  "NumCreditLines": 5,
  "InterestRate": 8.5,
  "LoanTerm": 36,
  "DTIRatio": 25.0,
  "Education": "Bachelor",
  "EmploymentType": "Salaried",
  "MaritalStatus": "Married",
  "HasMortgage": "No",
  "HasDependents": "Yes",
  "LoanPurpose": "Car",
  "HasCoSigner": "No"
}
```

**Example Response**:
```json
{
  "risk_score": 0.991,
  "features_importance": {
    "poly__Age": 0.5565,
    "poly__LoanAmount": 0.4898,
    "poly__InterestRate": 0.4897
  }
}
```

## Business Impact

### Risk Assessment Strategy

Our **recall-focused approach** delivers:

- **70% Default Detection**: Identifies 7 out of 10 potential defaults
- **Risk Mitigation**: Prevents ~70% of potential loan losses
- **Portfolio Protection**: Maintains healthier loan book
- **Regulatory Compliance**: Demonstrates robust risk capabilities

### Cost-Benefit Analysis

- **Prevented Losses**: ~70% of defaults caught = significant financial protection
- **Trade-off**: Lower precision (22%) is acceptable given cost asymmetry
- **ROI**: Cost of false positives << Cost of false negatives

## Technical Features

- **Multiple ML Models**: AdaBoost, LightGBM, Logistic Regression, Voting Classifier
- **Preprocessing Pipeline**: Categorical encoding, scaling, feature transformations
- **Feature Importance**: SHAP integration for model interpretability
- **Class Imbalance Handling**: Proper sampling and evaluation techniques
- **FastAPI Integration**: Production-ready REST API
- **Comprehensive EDA**: Jupyter notebooks for analysis and visualization

## Notebooks

- `eda_and_visualization.ipynb`: Complete exploratory data analysis

## Model Selection Rationale

**Logistic Regression** selected as production model because:

1. **Highest Recall (70.23%)**: Maximum default detection
2. **Interpretability**: Linear relationships easily explained
3. **Stability**: Consistent performance across datasets
4. **Speed**: Fast inference for real-time decisions
5. **Regulatory Friendly**: Transparent decision-making process

## Future Enhancements

- **Feature Engineering**: Polynomial features and interaction terms
- **Advanced Ensembles**: Stacking and blending techniques  
- **Threshold Optimization**: Dynamic threshold adjustment
- **Real-world Validation**: Production data testing
- **Monitoring**: Model drift detection and retraining pipelines

## Author

**Mithun MS**

---

**Note**: This project prioritizes recall over other metrics to minimize financial risk. The 70% recall achievement represents a significant improvement in loan default detection capabilities, providing substantial value to financial institutions through risk mitigation and portfolio protection.