# preprocessor.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Import the configuration variable
from .config import PREPROCESSOR_PATH

class LoanPreprocessor:
    """
    Handles all data transformation steps by loading and applying a
    pre-trained Scikit-learn preprocessing pipeline.
    """
    def __init__(self, pipeline_path: str = PREPROCESSOR_PATH):
        """Initializes by loading the fitted preprocessing pipeline."""
        try:
            print(f"Loading preprocessing pipeline from: {pipeline_path}")
            self.pipeline: Pipeline = joblib.load(pipeline_path)
            print("✅ Preprocessing pipeline loaded successfully.")
        except FileNotFoundError:
            print(f"❌ ERROR: Preprocessing pipeline not found at '{pipeline_path}'.")
            raise

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features from the raw input data."""
        df_engineered = df.copy()
        epsilon = 1e-6
        print("🔧 Engineering new features...")
        
        # Original engineered features
        df_engineered['Loan_to_Income_Ratio'] = df_engineered['LoanAmount'] / (df_engineered['Income'] + epsilon)
        df_engineered['Debt_to_Employment_Ratio'] = df_engineered['DTIRatio'] / (df_engineered['MonthsEmployed'] + 1)
        
        # --- FIX: Add the missing IncomePerCredit engineered feature ---
        df_engineered['IncomePerCredit'] = df_engineered['Income'] / (df_engineered['NumCreditLines'] + epsilon)
        
        return df_engineered

    def preprocess(self, new_data_df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full preprocessing workflow."""
        engineered_df = self._feature_engineer(new_data_df)
        transformed_data = self.pipeline.transform(engineered_df)
        feature_names = self.pipeline.get_feature_names_out()
        processed_df = pd.DataFrame(
            transformed_data, 
            columns=feature_names, 
            index=new_data_df.index
        )
        print("✅ Data preprocessing complete.")
        return processed_df
