# =========================================
# Spam Mail Detector using Machine Learning
# =========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Download Stopwords (SAFE METHOD)
try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')


# 3. Load Dataset
# Dataset: SMS Spam Collection (UCI)
# File name: spam.csv (must be in same folder)

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset Loaded Successfully\n")
print(df.head())


# 4. Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


df['clean_message'] = df['message'].apply(clean_text)


# 5. Encode Labels
# spam -> 1, ham -> 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# 6. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']


# 7. Train-Test Split (Improved)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)


# 9. Predictions
y_pred = model.predict(X_test)


# 10. Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 11. Test with New Message
new_message = ["Congratulations! You have won a free prize. Click now"]
new_message_clean = [clean_text(new_message[0])]
new_message_vector = vectorizer.transform(new_message_clean)

prediction = model.predict(new_message_vector)

print("\nNew Message Prediction:")
if prediction[0] == 1:
    print("SPAM 🚨")
else:
    print("HAM (Not Spam) ✅")
