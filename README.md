# Sports-Politics-Text-Classifier



## Overview

This project implements a machine learning pipeline to classify news documents into two categories:
- Sports
- Politics

The aim is to compare multiple feature representations and machine learning techniques for document classification.

---

## Dataset

A public news dataset was used and filtered into two categories:

- SPORTS: 5077 articles
- POLITICS: 35602 articles

Each document consists of headline and short description.

---

## Preprocessing

- Lowercasing
- Stopword removal
- Tokenization
- Noise removal

---

## Feature Engineering

Three feature extraction techniques were used:

1. Bag of Words
2. TF-IDF
3. N-grams (bi-grams)

---

## Machine Learning Models

The following models were trained and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Results Summary

| Feature | Model | Accuracy |
|--------|------|----------|
| BoW | Naive Bayes | 98.05% |
| BoW | Logistic Regression | 97.59% |
| BoW | SVM | 97.40% |
| TF-IDF | Naive Bayes | 92.72% |
| TF-IDF | Logistic Regression | 96.25% |
| TF-IDF | SVM | 97.75% |
| N-grams | Naive Bayes | 95.73% |
| N-grams | Logistic Regression | 97.52% |
| N-grams | SVM | 97.73% |

Best performing combination:
→ Bag of Words + Naive Bayes  
→ TF-IDF + SVM

---

## Observations

- Feature engineering plays a crucial role in text classification.
- SVM handles high dimensional sparse text data effectively.
- Naive Bayes performs strongly with Bag of Words representation.

---

## Limitations

- Dataset imbalance
- Vocabulary overlap
- Contextual ambiguity

---

## Future Work

- Deep learning models (BERT, LSTM)
- Multi-class classification
- Topic modelling

---

## Author

Roll Number: m23ma2001
Course: Natural Language Understanding
