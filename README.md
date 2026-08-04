# Sentiment Analyzer using Machine Learning and Embeddings

## Overview

Sentiment Analyzer is an AI-powered text classification system that analyzes user input (tweets, reviews, and comments) and classifies the sentiment as:

- Positive
- Negative

The project uses classical Machine Learning techniques (TF-IDF + Logistic Regression) and advanced NLP embeddings (BERT-based sentence embeddings) to compare performance and improve sentiment prediction.

The system also integrates PostgreSQL with SQLAlchemy and Alembic to store and retrieve prediction history.

---

# Project Goals

This project was developed to achieve the following:

## Day 1 — Classical Machine Learning Model

- Collect and explore sentiment data
- Clean and preprocess text data
- Extract features using TF-IDF Vectorization
- Train a Logistic Regression classifier
- Evaluate model performance using:

  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Classification Report

---

## Day 2 — Advanced NLP Features

- Generate text embeddings using Sentence Transformers (BERT-based embeddings)
- Compare TF-IDF features against embedding-based features
- Tune machine learning parameters
- Save trained models using Joblib
- Build a prediction pipeline for new text inputs
- Visualize and compare model performance

---

## Day 3 — Database Integration

- Design a sentiment prediction database
- Create SQLAlchemy models
- Manage database schema using Alembic migrations
- Store prediction history
- Retrieve previous predictions
- Connect the trained AI model with the database

---

# Technologies Used

## Programming Language

- Python 3

## Machine Learning

- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization

## Natural Language Processing

- Sentence Transformers
- BERT-based embeddings

## Database

- PostgreSQL
- SQLAlchemy ORM
- Alembic

## Data Processing

- Pandas
- NumPy

## Model Serialization

- Joblib

---

# Project Structure
