"""Loan Default Prediction Module.

This module provides a LoanDefaultPredictor class for predicting loan default probabilities
and analyzing feature importance using a trained machine learning model.
"""



import pandas as pd
import joblib
import json
import numpy as np
from typing import Any, Dict

from .config import PREPROCESSOR_PATH, MODEL_PATH
from .preprocessing import LoanPreprocessor

class LoanDefaultPredictor:
    """A loan default risk prediction system.
    
    This class loads a trained model and preprocessing pipeline to:
    - Predict default probabilities for loan applications
    - Analyze feature importance in predictions
    
    Args:
        preprocessor_path: Path to the saved preprocessing pipeline
        model_path: Path to the trained model file
        
    Attributes:
        preprocessor: LoanPreprocessor instance for data transformation
        model: Trained machine learning model
    """
    def __init__(self, preprocessor_path: str = PREPROCESSOR_PATH, model_path: str = MODEL_PATH):
        # Load preprocessor
        self.preprocessor = LoanPreprocessor(pipeline_path=preprocessor_path)
        
        # Load model
        try:
            print(f"Loading final model from: {model_path}")
            self.model = joblib.load(model_path)
            print("✅ Final model loaded successfully.")
        except FileNotFoundError:
            print(f"❌ ERROR: Model file not found at '{model_path}'.")
            raise

    def predict(self, df: pd.DataFrame) -> Any:
        """
        Returns probabilities of the positive class.
        """
        processed_df = self.preprocessor.preprocess(df)
        probabilities = self.model.predict_proba(processed_df)[:, 1]
        return probabilities
         
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Returns feature importance for the single model.
        """
        processed_df = self.preprocessor.preprocess(df)
        feature_names = processed_df.columns

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
        else:
            importances = np.zeros(len(feature_names))

        return dict(zip(feature_names, importances))

# =======================
# Main usage example
# =======================
if __name__ == "__main__":
    sample_json = """
    {
        "Age": 42,
        "Income": 85000,
        "LoanAmount": 25000,
        "CreditScore": 680,
        "MonthsEmployed": 72,
        "NumCreditLines": 6,
        "InterestRate": 11.5,
        "LoanTerm": 36,
        "DTIRatio": 0.4,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Debt Consolidation",
        "HasCoSigner": "No"
    }
    """

    try:
        # Initialize predictor
        predictor = LoanDefaultPredictor()
        
        # Convert JSON to DataFrame
        data_dict = json.loads(sample_json)
        df_input = pd.DataFrame([data_dict])

        # 1️⃣ Predict probability
        probs = predictor.predict(df_input)
        print("\n--- 🏁 Predicted Probability ---")
        print(f"Probability of Default: {probs[0]:.4f}")

        # 2️⃣ Get feature importance
        feat_imp = predictor.get_feature_importance(df_input)
        sorted_fi = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
        print("\n--- Top 5 Feature Importances ---")
        for k, v in list(sorted_fi.items())[:5]:
            print(f"{k}: {v:.4f}")

    except Exception as e:
        print(f"\n--- 💥 An error occurred: {e} ---")
