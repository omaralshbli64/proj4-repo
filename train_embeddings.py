import pandas as pd
import re
import joblib


from sentence_transformers import SentenceTransformer


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
    accuracy_score,
    classification_report
)



# Load data

df = pd.read_csv(
    "data/sent140_cleaned.csv"
)



def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    return text



df["text"] = df["text"].apply(
    clean_text
)



X = df["text"]

y = df["sentiment"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# BERT embedding model

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



X_train_emb = encoder.encode(
    X_train.tolist()
)


X_test_emb = encoder.encode(
    X_test.tolist()
)



# Train classifier

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_emb,
    y_train
)



y_pred = model.predict(
    X_test_emb
)



print(
    "Embedding Accuracy:"
)

print(
    accuracy_score(
        y_test,
        y_pred
    )
)



print(
    classification_report(
        y_test,
        y_pred
    )
)



joblib.dump(
    model,
    "models/embedding_model.pkl"
)


print(
    "Embedding model saved"
)