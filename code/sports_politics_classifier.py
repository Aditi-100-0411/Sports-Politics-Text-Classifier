import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""
first i am filtering the dataset and loading it 

"""

def load_and_filter_dataset(file_path):

    sports_texts = []
    politics_texts = []

    print("Reading dataset and filtering categories...")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)

            category = article["category"]
            text = article["headline"] + " " + article["short_description"]

            if category == "SPORTS":
                sports_texts.append(text)

            elif category == "POLITICS":
                politics_texts.append(text)
    print("Sports articles:", len(sports_texts))
    print("Politics articles:", len(politics_texts))

    texts = sports_texts + politics_texts
    labels = ["sports"] * len(sports_texts) + ["politics"] * len(politics_texts)

    return texts, labels



# Evaluation

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, pos_label="sports"))
    print("Recall:", recall_score(y_test, preds, pos_label="sports"))
    print("F1 Score:", f1_score(y_test, preds, pos_label="sports"))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("-" * 50)


# Main


def main():

    dataset_path = "News_Category_Dataset_v3 2.json"

    texts, labels = load_and_filter_dataset(dataset_path)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    
    # Feature 1: Bag of Words
    
    print("\n BAG OF WORDS")

    bow = CountVectorizer(stop_words="english")
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)

    # Naive Bayes
    print("Naive Bayes")
    nb = MultinomialNB()
    nb.fit(X_train_bow, y_train)
    evaluate(nb, X_test_bow, y_test)

    # Logistic Regression
    print("Logistic Regression")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_bow, y_train)
    evaluate(lr, X_test_bow, y_test)
    # SVM
    print("SVM")
    svm = LinearSVC()
    svm.fit(X_train_bow, y_train)
    evaluate(svm, X_test_bow, y_test)


    
    # Feature 2: TF-IDF
    
    print("\n TF-IDF")

    tfidf = TfidfVectorizer(stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Naive Bayes")
    nb.fit(X_train_tfidf, y_train)
    evaluate(nb, X_test_tfidf, y_test)

    print("Logistic Regression")
    lr.fit(X_train_tfidf, y_train)
    evaluate(lr, X_test_tfidf, y_test)

    print("SVM")
    svm.fit(X_train_tfidf, y_train)
    evaluate(svm, X_test_tfidf, y_test)


    
    # Feature 3: N-Grams
   
    print("\n N-GRAMS ")

    ngram = CountVectorizer(ngram_range=(1,2), stop_words="english")
    X_train_ng = ngram.fit_transform(X_train)
    X_test_ng = ngram.transform(X_test)
    print("Naive Bayes")
    nb.fit(X_train_ng, y_train)
    evaluate(nb, X_test_ng, y_test)

    print("Logistic Regression")
    lr.fit(X_train_ng, y_train)
    evaluate(lr, X_test_ng, y_test)

    print("SVM")
    svm.fit(X_train_ng, y_train)
    evaluate(svm, X_test_ng, y_test)


if __name__ == "__main__":
    main()
