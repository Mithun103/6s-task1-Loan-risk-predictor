# **LoanRiskPredictor**

A machine learning project designed to predict loan default risk with a focus on maximizing recall to minimize financial losses from undetected high-risk loans.

## **Project Overview**

LoanRiskPredictor utilizes multiple machine learning models, preprocessing pipelines, and feature engineering techniques to provide robust loan default predictions. The system prioritizes **recall over precision** to ensure maximum identification of potential defaults, protecting financial institutions from significant losses.

**Key Achievement**: 70.23% recall rate using a model trained on a specialized, business-oriented dataset, successfully identifying 7 out of every 10 loans that would default.

## **Dataset Overview**

* **Records**: 255,246 loan applications  
* **Features**: Numerical and categorical variables  
* **Target**: Default (0 \= No Default, 1 \= Default)  
* **Class Distribution**: Originally highly imbalanced (7.5:1 ratio) before being balanced for model training.

## **Repository Structure**

LoanRiskPredictor/  
│  
├── api/  
│   ├── \_\_init\_\_.py  
│   └── fastapi\_app.py              \# FastAPI application to serve ML predictions  
│  
├── data/                           \# Raw and processed datasets  
│  
├── models(business)/               \# Models trained on a specialized, non-leakable dataset  
│   ├── adaboost\_model.joblib  
│   ├── lightgbm\_model.joblib  
│   ├── logistic\_regression.joblib  
│   └── preprocessing\_pipeline.joblib  
│  
├── models(validation)/             \# High-performance models for testing the dataset  
│   ├── lightgbm(validation).joblib  
│   └── xgboost(validation).joblib  
│  
├── eda\_and\_training.ipynb          \# Jupyter notebooks for EDA and training  
│  
├── src/                            \# Source code for ML pipeline  
│   ├── \_\_init\_\_.py  
│   ├── config.py                   \# Paths and configuration constants  
│   ├── inference.py                \# ML inference and feature importance  
│   └── preprocessing.py            \# Data preprocessing scripts  
│  
├── requirements.txt                \# Python dependencies  
└── README.md                       \# Project documentation

## **Model Usage Guide: Understanding the Model Sets**

This project features two distinct sets of models. While both are trained on balanced datasets, they are designed for different strategic purposes:

* ### **Business Models (models(business)/)**

  * **Purpose**: These models are trained on a **specially crafted, non-leakable balanced dataset**. This dataset was carefully constructed to align with the primary business objective: maximizing recall to robustly identify potential defaults and minimize financial risk.  
  * **Use Case**: This is the **production-ready** set of models. The API exclusively uses the Logistic Regression model from this group because of its proven effectiveness in achieving the core business goal.

* ### **Validation Models (models(validation)/)**

  * **Purpose**: These models are trained on a standard balanced dataset to **benchmark maximum achievable performance** under ideal conditions.  
  * **Use Case**: Use these models if you want to **test the data**. They are perfect for comparative analysis or to understand the upper limits of predictive power on a clean, perfectly balanced dataset.

## **Model Performance (Business Objective)**

This section details the performance of the **Business Models**, which are optimized for the primary goal of high recall.

### **Why Recall Matters in Risk Prediction**

For the primary business problem, **recall is our main metric** because:

* **Cost Asymmetry**: Missing a default costs the entire loan amount (\~$15,000 average), while rejecting a good applicant only costs potential interest revenue (\~$2,000).  
* **Risk Management**: Financial institutions must identify high-risk loans to protect portfolio health.  
* **Business Impact**: 70% recall translates to preventing \~70% of potential loan losses.

### **Model Comparison (Business-Objective Data)**

| Model | Recall | Precision | F1-Score | ROC AUC |
| :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression** ⭐ | **70.23%** | 22.31% | 33.86% | 75.63% |
| AdaBoost | **69.10%** | 22.59% | 34.05% | 75.70% |
| LightGBM | **66.89%** | 23.00% | 34.00% | \- |

**Selected Model**: **Logistic Regression** (highest recall for the business problem).

## **Validation Model Performance **

The following **Validation Models** were trained on a standard balanced dataset to benchmark maximum achievable performance under ideal conditions.

### **Validation Model Comparison**

| Model | Recall | Precision | F1-Score |
| :---- | :---- | :---- | :---- |
| **XGBoost** ⭐ | 99.82% | 98.03% | 98.92% |
| LightGBM | 99.12% | 95.75% | 97.41% |

## **Quick Start**

### **Installation**

1. **Clone the repository**:  
   git clone https://github.com/Mithun103/6s-task1-Loan-risk-predictor.git 
   cd LoanRiskPredictor

2. **Create virtual environment**:  
   python \-m venv .venv  
   source .venv/bin/activate      \# Linux/Mac  
   .venv\\Scripts\\activate         \# Windows

3. **Install dependencies**:  
   pip install \-r requirements.txt

### **Running the API**

From the project's root directory, start the FastAPI server:

uvicorn LoanRiskPredictor.api.fastapi\_app:app \--reload

Access interactive documentation at http://127.0.0.1:8000/docs.

## **API Usage and Model Selection**

The provided API is specifically developed from a **business perspective**, using a model trained on a specially crafted dataset to maximize recall. Therefore, the API exclusively uses the **Logistic Regression** model from the models(business)/ directory, which best achieved this primary goal.

If you want to **test the data**'s performance on a standard balanced dataset, you can download and use any of the high-performance models from the models(validation)/ directory.

### **API Endpoints**

#### **GET /**

Returns a welcome message.

#### **GET /health**

Checks if the predictor model is loaded and ready.

**Example Response**:

{  
  "predictor\_status": "loaded"  
}

#### **POST /predict**

Predicts loan default risk for a single applicant and returns the feature importance.

**Example Request**:

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

#### **GET /metrics**

Returns the final evaluation metrics for the production model (Logistic Regression).

**Example Response**:

{  
  "classification\_report": {  
    "0": { "precision": 0.95, "recall": 0.68, "f1\_score": 0.79, "support": 45121 },  
    "1": { "precision": 0.22, "recall": 0.70, "f1\_score": 0.34, "support": 5929 }  
  },  
  "confusion\_matrix": \[  
    \[30622, 14499\],  
    \[1765, 4164\]  
  \],  
  "roc\_auc\_score": 0.75  
}

## **Future Enhancements**

* **Advanced Ensembles**: Stacking and blending techniques.  
* **Threshold Optimization**: Dynamic threshold adjustment for the business model.  
* **Real-world Validation**: Production A/B testing.  
* **Monitoring**: Implement model drift detection and retraining pipelines.

## **Author**

**Mithun MS**