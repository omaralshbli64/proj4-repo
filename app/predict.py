import joblib


from .db import SessionLocal


from .models import SentimentPrediction



model = joblib.load(
    "models/tfidf_model.pkl"
)


vectorizer = joblib.load(
    "models/vectorizer.pkl"
)




def predict(text):


    data = vectorizer.transform(
        [text]
    )


    sentiment = model.predict(
        data
    )[0]


    score = float(
    max(model.predict_proba(data)[0])
    ) 


    save_prediction(
        text,
        sentiment,
        score
    )


    return sentiment, score





def save_prediction(
    text,
    sentiment,
    score
):

    db = SessionLocal()


    prediction = SentimentPrediction(

        input_text=text,

        predicted_sentiment=sentiment,

        prediction_score=score

    )


    db.add(prediction)


    db.commit()


    db.close()




def get_all_predictions():

    db = SessionLocal()


    results = db.query(
        SentimentPrediction
    ).all()


    db.close()


    return results