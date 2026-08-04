from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from datetime import datetime


from .db import Base



class SentimentPrediction(Base):

    __tablename__ = "sentiment_predictions"



    id = Column(
        Integer,
        primary_key=True
    )


    input_text = Column(
        String,
        nullable=False
    )


    predicted_sentiment = Column(
        String,
        nullable=False
    )


    prediction_score = Column(
        Float
    )


    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )