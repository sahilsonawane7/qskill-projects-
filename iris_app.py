# =========================================
# Iris Flower Classification Web App
# =========================================

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# App title
st.title("🌸 Iris Flower Classification Web App")
st.write("Enter flower measurements to predict the species")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model (Logistic Regression + Scaling)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])

model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.1
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 2.0, 4.5, 3.5
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", 1.0, 7.0, 1.4
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", 0.1, 2.5, 0.2
)

# Input array
input_data = np.array([
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]).reshape(1, -1)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Output
st.subheader("🌼 Prediction Result")
st.write("**Predicted Species:**", iris.target_names[prediction][0])

st.subheader("📊 Prediction Probabilities")
for name, prob in zip(iris.target_names, prediction_proba[0]):
    st.write(f"{name}: {prob:.2f}")
