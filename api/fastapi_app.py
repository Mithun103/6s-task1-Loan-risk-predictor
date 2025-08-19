"""
Loan Default Prediction API

This module provides a FastAPI application for predicting loan default risk
based on applicant features. It loads a pre-trained ML model and preprocessing
pipeline from the `src` package, provides endpoints to predict risk scores, 
and returns feature importance explanations.

Endpoints:
- GET /         : Returns a welcome message.
- POST /predict : Accepts a loan application JSON and returns the predicted
                  risk score and sorted feature importance.
- GET /health   : Returns the status of the model loading.

Author: Mithun MS
Version: 1.0.2
"""

import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

# -----------------------------
# Add LoanRiskPredictor root to Python path
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

# -----------------------------
# Module Imports
# -----------------------------
try:
    from src.config import PREPROCESSOR_PATH, MODEL_PATH
    from src.inference import LoanDefaultPredictor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Required module not found: {e}. "
                              f"Check that 'src' folder exists and contains '__init__.py'") from e

# -----------------------------
# Pydantic Model for Input Data
# -----------------------------
class LoanApplication(BaseModel):
    """Schema for loan application input data."""
    Age: int
    Income: int
    LoanAmount: int
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="API to predict loan default and provide feature importance.",
    version="1.0.2"
)

# -----------------------------
# Global variable for predictor
# -----------------------------
predictor: LoanDefaultPredictor = None

# -----------------------------
# Startup Event: Load ML Modules
# -----------------------------
@app.on_event("startup")
def load_modules():
    """Load the ML predictor on application startup."""
    global predictor

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not Path(PREPROCESSOR_PATH).exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")

    predictor = LoanDefaultPredictor(
        preprocessor_path=PREPROCESSOR_PATH,
        model_path=MODEL_PATH
    )
    print("--- Inference Module Loaded Successfully ---")

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def read_root():
    """Return welcome message."""
    return {"message": "Welcome to the Loan Default Prediction API. Go to /docs for interactive docs."}

# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health_check():
    """Check if the predictor is loaded."""
    status = "loaded" if predictor else "not loaded"
    return {"predictor_status": status}

@app.get("/metrics")
def get_metrics():
    """Return final model evaluation metrics."""
    metrics = {
        "classification_report": {
            "0": {"precision": 0.95, "recall": 0.68, "f1_score": 0.79, "support": 45121},
            "1": {"precision": 0.22, "recall": 0.70, "f1_score": 0.34, "support": 5929},
            "accuracy": 0.68,
            "macro_avg": {"precision": 0.58, "recall": 0.69, "f1_score": 0.56, "support": 51050},
            "weighted_avg": {"precision": 0.86, "recall": 0.68, "f1_score": 0.74, "support": 51050}
        },
        "confusion_matrix": [
            [30622, 14499],
            [1765, 4164]
        ],
        "roc_auc_score": 0.75
    }
    return metrics


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(application: LoanApplication):
    """
    Predict loan default risk and return feature importance.

    Args:
        application (LoanApplication): JSON input for loan application features.

    Returns:
        dict: Contains predicted risk score and sorted feature importance.

    Raises:
        HTTPException: If prediction fails or predictor is not loaded.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model modules are not loaded. Please wait a moment and try again."
        )

    try:
        df_input = pd.DataFrame([application.dict()])

        pred_prob = predictor.predict(df_input)[0]

        # 2️⃣ Get feature importance
        feat_imp = predictor.get_feature_importance(df_input)
        sorted_fi = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        return {
            "risk_score": float(pred_prob),
            "features_importance": sorted_fi
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )

