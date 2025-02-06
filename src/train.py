import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocessor import preprocess_data

# Load dataset
df = pd.read_csv("../data/car_prices.csv")
df = df.dropna()

# Preprocess dataset
df = preprocess_data(df, training=True)

# Define features (X) and target variable (y)
X = df[['Year', 'KM_Driven', 'Brand']]
y = df['Selling_Price']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Save the trained model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
