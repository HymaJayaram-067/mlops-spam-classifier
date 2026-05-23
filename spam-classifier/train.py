import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Simple training data
data = {
    "text": [
        "Win free money now click here",
        "You won a lottery claim your prize",
        "Free offer limited time buy now",
        "Hey are you coming to lunch today",
        "Meeting at 3pm in conference room",
        "Please review the attached document"
    ],
    "label": [1, 1, 1, 0, 0, 0]  # 1=spam, 0=not spam
}

df = pd.DataFrame(data)

# Train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultinomialNB())
])

pipeline.fit(df['text'], df['label'])

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved as model.pkl")
