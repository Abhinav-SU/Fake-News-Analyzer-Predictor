import argparse
import os
import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_datasets(dataset_dir):
    fake_path = os.path.join(dataset_dir, "FakeNewsData.csv")
    true_path = os.path.join(dataset_dir, "TrueNewsData.csv")
    data_fake = pd.read_csv(fake_path)
    data_true = pd.read_csv(true_path)
    data_fake["class"] = 0
    data_true["class"] = 1
    data = pd.concat([data_fake, data_true], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)
    data.drop(["title", "subject", "date"], axis=1, inplace=True)
    return data


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+|[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def train_and_evaluate(data):
    data["text"] = data["text"].apply(preprocess_text)
    X = data["text"]
    y = data["class"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=0),
    }

    for name, model in models.items():
        model.fit(xv_train, y_train)
        preds = model.predict(xv_test)
        print(f"\n{name} Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
        print(classification_report(y_test, preds))


def main():
    parser = argparse.ArgumentParser(description="Train fake news detection models")
    parser.add_argument(
        "--dataset_dir",
        default="datasets",
        help="Directory containing FakeNewsData.csv and TrueNewsData.csv",
    )
    args = parser.parse_args()

    data = load_datasets(args.dataset_dir)
    train_and_evaluate(data)


if __name__ == "__main__":
    main()
