import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_train_model():
    response = client.post("/train/")
    assert response.status_code == 204

def test_predict():
    data = {"data": [[0.038075906, 0.05068012, 0.061696206, 0.021872354, -0.044223498, -0.03482076, -0.043400846, -0.002592261, 0.01990749, -0.017646125]]}
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()[0]
    assert "probabilities" in response.json()[0]

def test_invalid_data():
    data = {"data": [[1, 2, 3]]}
    response = client.post("/predict/", json=data)
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Each input data point must have 10 features" in response.json()["detail"]

def test_invalid_endpoint():
    response = client.get("/invalid_endpoint/")
    assert response.status_code == 404