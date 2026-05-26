import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import mlflow

data = {
    "text": [
        "Win free money now click here",
        "You won a lottery claim your prize",
        "Free offer limited time buy now",
        "Hey are you coming to lunch today",
        "Meeting at 3pm in conference room",
        "Please review the attached document"
    ],
    "label": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

for alpha in [0.1, 0.5, 1.0]:
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("model_type", "MultinomialNB")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('model', MultinomialNB(alpha=alpha))
        ])
        pipeline.fit(df['text'], df['label'])
        acc = accuracy_score(df['label'], pipeline.predict(df['text']))
        mlflow.log_metric("accuracy", acc)
        print(f"alpha={alpha} → accuracy={acc:.2f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved as model.pkl")
