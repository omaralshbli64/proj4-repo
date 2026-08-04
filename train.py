import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/sent140_cleaned.csv"
)


print(df.head())
print(df.columns)



# =========================
# CLEAN TEXT
# =========================


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text



df["text"] = df["text"].apply(
    clean_text
)



# =========================
# SELECT FEATURES
# =========================


X = df["text"]

y = df["sentiment"]



# =========================
# SPLIT DATA
# =========================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# =========================
# TF-IDF
# =========================


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)



# =========================
# TRAIN MODEL
# =========================


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_tfidf,
    y_train
)



# =========================
# EVALUATION
# =========================


y_pred = model.predict(
    X_test_tfidf
)


print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)


print("\nPrecision:")
print(
    precision_score(
        y_test,
        y_pred,
        average="weighted"
    )
)


print("\nRecall:")
print(
    recall_score(
        y_test,
        y_pred,
        average="weighted"
    )
)


print("\nF1 Score:")
print(
    f1_score(
        y_test,
        y_pred,
        average="weighted"
    )
)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



# =========================
# SAVE MODEL
# =========================


joblib.dump(
    model,
    "models/tfidf_model.pkl"
)


joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)


print("\nTF-IDF model saved")