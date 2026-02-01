# ================================
# Iris Flower Classification (Advanced)
# ================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y

print("Dataset Preview:")
print(df.head())


# 3. Data Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df["petal length (cm)"],
    y=df["petal width (cm)"],
    hue=df["species"]
)
plt.title("Petal Length vs Petal Width")
plt.show()


# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Build Pipelines
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200))
    ]),
    
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    
    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ])
}


# 6. Train & Evaluate Models
print("\nModel Accuracies:")
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")


# 7. Cross-Validation (Best Practice)
log_reg_cv = cross_val_score(
    pipelines["Logistic Regression"], X, y, cv=5
)
print("\nLogistic Regression CV Accuracy:", log_reg_cv.mean())


# 8. Confusion Matrix (Best Model)
best_model = pipelines["Logistic Regression"]
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 9. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))


# 10. Advanced Prediction with Probabilities
sample_flower = [[6.1, 2.8, 4.7, 1.2]]

prediction = best_model.predict(sample_flower)
probability = best_model.predict_proba(sample_flower)

print("\nNew Flower Prediction:")
print("Predicted Species:", iris.target_names[prediction][0])

print("\nPrediction Probabilities:")
for name, prob in zip(iris.target_names, probability[0]):
    print(f"{name}: {prob:.2f}")

