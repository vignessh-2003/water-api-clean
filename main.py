import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from fastapi import Request

# ------------- 1) Define your FastAPI app -------------
app = FastAPI(
    title="Water Potability Predictor (5-Feature Model)",
    version="2.0"
)

# ------------- ADD THIS SECTION FOR CORS CONFIGURATION -------------
# Define allowed origins for your frontend.
# IMPORTANT:
# - For local development, include "http://localhost:3000" (or your React dev server port)
# - For this Canvas environment, include its origin (e.g., https://<your_unique_id>.scf.usercontent.goog)
# - For your deployed React app, include its production URL (e.g., https://your-deployed-app.vercel.app)
# For testing, you can use "*" to allow all origins, but this is NOT recommended for production.
origins = [
    "http://localhost:3000",
    "https://56pyji0hkj8g0jgq4fdnxulmmcvagztw397d5wrbibfmqgo37k-h769614562.scf.usercontent.goog", # Example Canvas origin, check your browser
    "https://0s1c4fphwxba6dyevhj0rpyz2ybppkgsh9xqqvvyn0rv5c8mzi-h769614562.scf.usercontent.goog", # Another example Canvas origin
    "https://*.scf.usercontent.goog", # A wildcard for scf.usercontent.goog to cover Canvas variations (use with caution)
    # Add your deployed React app's production URL here, e.g.:
    # "https://your-react-app-domain.com",
    # "https://your-react-app-name.vercel.app",
    # "https://your-react-app-name.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allow cookies/authentication headers
    allow_methods=["*"],    # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all headers
)
# ------------- END CORS CONFIGURATION SECTION -------------


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
    # Ensure these paths are correct relative to your main.py on Railway
    model = joblib.load("xgb_baseline_model.pkl")
    scaler = joblib.load("xgb_baseline_scaler.save")
    print("[INFO] Loaded model and scaler for 5-feature pipeline.")
except Exception as e:
    print("[ERROR] Could not load model or scaler:", e)
    raise HTTPException(status_code=500, detail=f"Model or scaler not loaded: {e}") # Raise HTTP exception

# ------------- 5) Predict endpoint -------------
@app.post("/predict", summary="Predict water potability from 5 sensor readings")
def predict(input: SensorInput):
    # Ensure the feature order matches what your model expects
    features = [
        input.ph,
        input.Temperature,
        input.DissolvedOxygen,
        input.TDS,
        input.Turbidity
    ]
    try:
        X_scaled = scaler.transform([features])
        pred = model.predict(X_scaled)[0]
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(err)}") # More specific error

    # FastAPI automatically serializes this to JSON
    return {
        "prediction": "Potable" if pred == 1 else "Not Potable", # Return string directly
        "result": "Potable" if pred == 1 else "Not Potable"
    }

@app.post("/sensordata", summary="Receive raw sensor data from AWS IoT")
async def receive_sensor_data(request: Request):
    payload = await request.json()
    print("[AWS IoT] Received Payload:", payload)
    return {"message": "Sensor data received"}
# ------------- 6) Run with Uvicorn -------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # It's good practice to set reload=False for production deployments
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
