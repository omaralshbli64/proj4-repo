from app.predict import predict



texts = [

"I love this product, it works perfectly!",

"This was a terrible experience. I will never buy it again.",

"The service was amazing"

]



for text in texts:


    sentiment, score = predict(text)


    print("----------------------------")


    print(
        "Text:",
        text
    )


    print(
        "Predicted Sentiment:",
        sentiment
    )


    print(
        "Score:",
        score
    )