import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, training=True):
    """
    Preprocesses the dataset for training and inference.
    If training=True, it fits LabelEncoders and Scaler. Otherwise, it loads them.
    """
    label_encoders = {}
    scaler = StandardScaler()

    # Load pre-fitted encoders and scalers for inference
    if not training:
        label_encoders = joblib.load("label_encoders.pkl")
        scaler = joblib.load("scaler.pkl")

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        if training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            df[col] = label_encoders[col].transform(df[col])

    # Feature scaling
    scaled_columns = ['KM_Driven', 'Year']
    if training:
        df[scaled_columns] = scaler.fit_transform(df[scaled_columns])
        joblib.dump(scaler, "scaler.pkl")  # Save fitted scaler
        joblib.dump(label_encoders, "label_encoders.pkl")  # Save fitted encoders
    else:
        df[scaled_columns] = scaler.transform(df[scaled_columns])

    return df

# Example usage (for testing)
if __name__ == "__main__":
    df = pd.read_csv("../data/car_prices.csv")
    df = df.dropna()
    df = preprocess_data(df, training=True)
    df.to_csv("../data/car_prices_preprocessed.csv", index=False)
