from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import pickle
import json

app = FastAPI()

Instrumentator().instrument(app).expose(app)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

try:
    import redis
    r = redis.Redis(host="redis", port=6379, decode_responses=True, socket_connect_timeout=1)
    r.ping()
except Exception:
    r = None

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    pred = model.predict([input.text])[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    if r:
        try:
            r.lpush("predictions", json.dumps({"text": input.text, "prediction": label}))
        except Exception:
            pass
    return {"text": input.text, "prediction": label}

@app.get("/history")
def history():
    if not r:
        return []
    try:
        predictions = r.lrange("predictions", 0, 9)
        return [json.loads(p) for p in predictions]
    except Exception:
        return []

@app.get("/health")
def health():
    return {"status": "ok"}
