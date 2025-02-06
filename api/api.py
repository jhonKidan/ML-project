from fastapi import FastAPI
import joblib
import numpy as np

# Load model
model = joblib.load("src/model.pkl")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Car Price Prediction API"}

@app.post("/predict/")
def predict(year: int, mileage: float, brand_encoded: int):
    features = np.array([[year, mileage, brand_encoded]])
    prediction = model.predict(features)[0]
    return {"predicted_price": round(prediction, 2)}

# Run using: uvicorn app:app --reload
