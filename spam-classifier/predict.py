import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

test_texts = [
    "Click here to win a prize",
    "Can we reschedule our meeting?",
    "Free gift card waiting for you"
]

for text in test_texts:
    pred = model.predict([text])[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"'{text}' → {label}")
