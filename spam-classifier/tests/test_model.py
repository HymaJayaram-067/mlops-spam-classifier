import pytest
import pickle
import os

def test_model_file_exists():
    assert os.path.exists("model.pkl"), "model.pkl not found — run train.py first"

def test_model_predicts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict(["free money win prize now"])
    assert result[0] in [0, 1], f"Expected 0 or 1, got {result[0]}"

def test_model_not_spam():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict(["hello how are you today"])
    assert result[0] == 0, f"Expected 0 (ham), got {result[0]}"
