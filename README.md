# 🚗 Car Price Prediction using Linear Regression

## 📌 Project Overview
This project predicts the price of used cars based on features such as year, mileage, and brand.

## 📂 Folder Structure
- `data/` → Dataset files
- `notebooks/` → Jupyter notebooks for EDA, preprocessing, and training
- `src/` → Main Python scripts for training and prediction
- `api/` → FastAPI-based deployment script
- `README.md` → Project documentation

## 🚀 Getting Started
### 1️⃣ Install Dependencies

pip install -r requirements.txt


### 2️⃣ Train Model
python src/train.py


### 3️⃣ Run API
cd api uvicorn app:app --reload


## 📊 Results & Evaluation
The model was evaluated using Mean Squared Error (MSE) and R² Score.

## 🔗 Deployment
- Local API: `http://127.0.0.1:8000/predict/?year=2018&mileage=40000&brand_encoded=3`
