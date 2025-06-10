import pickle
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# --- API Setup ---
# Create a FastAPI app instance
app = FastAPI(title="Project Aether: Real-Time Fraud Detection API", version="1.0")

# --- Loading the Model ---
# Build a robust path to the model file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'model.pkl')

# Load the trained XGBoost model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully.")

# --- Pydantic Data Model for Input Validation ---
# This defines the structure and data types for the API request body.
# FastAPI uses this for automatic validation.
# The feature names (V1, V2, etc.) are based on the creditcard.csv dataset.
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Fraud Detection API. Go to /docs for usage."}

@app.post("/predict", tags=["Prediction"])
def predict_fraud(transaction: Transaction):
    """
    Predicts the probability of a transaction being fraudulent.
    Accepts a JSON payload with transaction features.
    """
    # 1. Convert the incoming Pydantic model to a pandas DataFrame
    input_df = pd.DataFrame([transaction.model_dump()])

    # 2. Replicate the preprocessing from the training script
    # IMPORTANT: Use a new scaler. In a real-world scenario, you would save and
    # load the scaler from the training phase to ensure consistency.
    scaler = StandardScaler()
    input_df['scaled_amount'] = scaler.fit_transform(input_df['Amount'].values.reshape(-1, 1))
    input_df['scaled_time'] = scaler.fit_transform(input_df['Time'].values.reshape(-1, 1))
    
    # Drop original columns and reorder to match training data
    input_df = input_df.drop(['Time', 'Amount'], axis=1)
    
    # The 'Class' column is not in the input, so we get the feature order from the model
    # Note: model.feature_names_in_ is an attribute from scikit-learn compatible models
    feature_order = model.feature_names_in_
    input_df = input_df[feature_order]

    # 3. Make the prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # 4. Format the response
    is_fraud = bool(prediction == 1)
    fraud_probability = float(prediction_proba[1])

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 4)
    }