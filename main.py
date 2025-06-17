from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
try:
    model = joblib.load("xgb_full_model.pkl")
    scaler = joblib.load("xgb_scaler.save")
    print("[INFO] Model and scaler loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load model or scaler:", e)

app = FastAPI()

class SensorInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.post("/predict")
def predict_water_quality(data: SensorInput):
    try:
        features = [
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity
        ]

        print("[INFO] Raw input:", features)

       
        X = scaler.transform([features])
        prediction = model.predict(X)[0]
        result = "Potable" if prediction == 1 else "Not Potable"

        return {
            "prediction": int(prediction),
            "result": result
        }

    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return {
            "error": str(e)
        }
