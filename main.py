# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Define the request body schema with exactly 5 features
class SensorInput(BaseModel):
    ph:              float = Field(..., example=7.0, description="pH level")
    Temperature:     float = Field(..., example=25.0, description="Water temperature (°C)")
    DissolvedOxygen: float = Field(..., example=8.5, description="Dissolved oxygen (mg/L)")
    TDS:             float = Field(..., example=300.0, description="Total dissolved solids (mg/L)")
    Turbidity:       float = Field(..., example=1.2, description="Turbidity (NTU)")

app = FastAPI(
    title="Water Potability Predictor (5-Feature Model)",
    version="2.0"
)

# Load your retrained model and scaler
try:
    model  = joblib.load("xgb_baseline_model.pkl")
    scaler = joblib.load("xgb_baseline_scaler.save")
    print("[INFO] Loaded model and scaler for 5-feature pipeline.")
except Exception as e:
    # If loading fails, crash early so you notice
    print("[ERROR] Could not load model or scaler:", e)
    raise

@app.post("/predict", summary="Predict water potability from 5 sensor readings")
def predict(input: SensorInput):
    """
    Returns:
    - prediction: 1 if Potable, 0 if Not Potable
    - result: "Potable" or "Not Potable"
    
    Example request body:
    {
      "ph": 7.1,
      "Temperature": 24.5,
      "DissolvedOxygen": 8.0,
      "TDS": 350.0,
      "Turbidity": 1.5
    }
    """
    # Build feature vector in the order the model expects
    features = [
        input.ph,
        input.Temperature,
        input.DissolvedOxygen,
        input.TDS,
        input.Turbidity
    ]

    try:
        # Scale and predict
        X_scaled = scaler.transform([features])  # shape (1,5)
        pred      = model.predict(X_scaled)[0]   # 0 or 1
    except Exception as err:
        # If something goes wrong, return a 500 with the error message
        raise HTTPException(status_code=500, detail=str(err))

    return {
        "prediction": int(pred),
        "result":     "Potable" if pred == 1 else "Not Potable"
    }
