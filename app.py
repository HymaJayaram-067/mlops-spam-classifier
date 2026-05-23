from flask import Flask, request, jsonify
import pickle
import redis
import json

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Connect to Redis
r = redis.Redis(host='redis', port=6379, decode_responses=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    pred = model.predict([text])[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    
    # Save to Redis
    r.lpush('predictions', json.dumps({"text": text, "prediction": label}))
    
    return jsonify({"text": text, "prediction": label})

@app.route('/history', methods=['GET'])
def history():
    predictions = r.lrange('predictions', 0, 9)  # last 10
    return jsonify([json.loads(p) for p in predictions])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
