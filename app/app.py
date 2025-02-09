import streamlit as st
import requests

# Streamlit App Title
st.title("🚗 Car Price Prediction App")

# User Input Fields
st.sidebar.header("Enter Car Details")
year = st.sidebar.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2018, step=1)
mileage = st.sidebar.number_input("Mileage (in km)", min_value=0, max_value=500000, value=40000, step=1000)
brand_encoded = st.sidebar.number_input("Brand Encoded", min_value=0, max_value=10, value=3, step=1)

# API URL (Ensure your FastAPI server is running)
API_URL = "https://ml-project-nbml.onrender.com/predict/"

# Button to Trigger Prediction
if st.sidebar.button("Predict Price"):
    # Send Request to FastAPI
    payload = {"year": year, "mileage": mileage, "brand_encoded": brand_encoded}
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        prediction = response.json().get("predicted_price", "Error")
        st.success(f"💰 Estimated Car Price: **${prediction}**")
    else:
        st.error("Error in fetching prediction. Ensure API is running.")

# Instructions
st.write("🔹 **How to Use:** Enter the car details on the left panel and click **Predict Price**.")
