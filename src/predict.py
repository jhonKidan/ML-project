import pandas as pd
import joblib
import sys
from preprocessor import preprocess_data

# Load the trained model
model = joblib.load("model.pkl")

# Example input data (can be replaced with user input)
input_data = {
    "Year": [2018],
    "KM_Driven": [45000],
    "Brand": ["Toyota"]
}

# Convert input to DataFrame
df_input = pd.DataFrame(input_data)

# Preprocess input data (set training=False)
df_input = preprocess_data(df_input, training=False)

# Make prediction
predicted_price = model.predict(df_input)

print(f"Predicted Car Price: ${predicted_price[0]:.2f}")
