# main.py

import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------- 1) Define your FastAPI app -------------
app = FastAPI(
    title="Water Potability Predictor (5-Feature Model)",
    version="2.0"
)

# ------------- 2) Health‐check endpoint -------------
@app.get("/", summary="Health check")
def health():
    """
    Quick ping to prove the server is alive.
    """
    return {"status": "ok"}

# ------------- 3) SensorInput schema -------------
class SensorInput(BaseModel):
    ph: float = Field(..., example=7.0, description="pH level")
    Temperature: float = Field(..., example=25.0, description="Water temperature (°C)")
    DissolvedOxygen: float = Field(..., example=8.5, description="Dissolved oxygen (mg/L)")
    TDS: float = Field(..., example=300.0, description="Total dissolved solids (mg/L)")
    Turbidity: float = Field(..., example=1.2, description="Turbidity (NTU)")

# ------------- 4) Load model & scaler -------------
try:
    model  = joblib.load("xgb_baseline_model.pkl")
    scaler = joblib.load("xgb_baseline_scaler.save")
    print("[INFO] Loaded model and scaler for 5-feature pipeline.")
except Exception as e:
    print("[ERROR] Could not load model or scaler:", e)
    raise

# ------------- 5) Predict endpoint -------------
@app.post("/predict", summary="Predict water potability from 5 sensor readings")
def predict(input: SensorInput):
    features = [
        input.ph,
        input.Temperature,
        input.DissolvedOxygen,
        input.TDS,
        input.Turbidity
    ]
    try:
        X_scaled = scaler.transform([features])
        pred      = model.predict(X_scaled)[0]
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    return {
        "prediction": int(pred),
        "result":     "Potable" if pred == 1 else "Not Potable"
    }

# ------------- 6) Run with Uvicorn -------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
