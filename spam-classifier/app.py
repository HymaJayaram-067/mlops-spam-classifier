from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import redis
import json

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

try:
    r = redis.Redis(host='redis', port=6379, decode_responses=True)
except:
    r = None

class TextInput(BaseModel):
    text: str

@app.post('/predict')
def predict(input: TextInput):
    pred = model.predict([input.text])[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    if r:
        r.lpush('predictions', json.dumps({"text": input.text, "prediction": label}))
    return {"text": input.text, "prediction": label}

@app.get('/history')
def history():
    if not r:
        return []
    predictions = r.lrange('predictions', 0, 9)
    return [json.loads(p) for p in predictions]

@app.get('/health')
def health():
    return {"status": "ok"}
