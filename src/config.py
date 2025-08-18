"""Configuration file for the Loan Risk Predictor project.

This module contains all path configurations and constants used throughout the project.
Paths are configured to work across different operating systems.
"""
from pathlib import Path

# --- DIRECTORY CONFIGURATION ---
# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Define paths for artifacts (models, pipelines, etc.)
ARTIFACTS_DIR = BASE_DIR / "models"

# --- FILE PATH CONFIGURATION ---
# Create the full paths for your artifacts by joining the directory and file name
PREPROCESSOR_PATH = "C:\\Mithunnn\\6S-TASK-1\\LoanRiskPredictor\\models\\preprocessing_pipeline.pkl"
MODEL_PATH = "C:\\Mithunnn\\6S-TASK-1\\LoanRiskPredictor\\models\\logistic_regression_model.joblib"
# You can add other configurations here as needed, for example:
# DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
