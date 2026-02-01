# =========================================
# House Price Prediction (One File)
# =========================================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Load Dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# IMPORTANT: Convert target to numeric
df["MEDV"] = pd.to_numeric(df["MEDV"])

# (Boston dataset has no missing values, so no fillna needed)

# 2. Features & Target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Convert all features to numeric (VERY IMPORTANT FIX)
X = X.apply(pd.to_numeric)


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# 6. Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)


# 7. Save Model & Scaler
joblib.dump(model, "house_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")


# 8. Load Model & Scaler
loaded_model = joblib.load("house_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")


# 9. Predict New House
sample_house = np.array([
    0.02, 0.0, 7.0, 0.0, 0.45,
    6.5, 60.0, 4.0, 2.0,
    300.0, 17.0, 390.0, 5.0
]).reshape(1, -1)

sample_house_scaled = loaded_scaler.transform(sample_house)
prediction = loaded_model.predict(sample_house_scaled)

print("Predicted House Price:", prediction[0])
