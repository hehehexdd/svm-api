import pytest
from app.model import load_model, get_prediction

@pytest.fixture
def trained_model():
    model, scaler = load_model()
    return model, scaler

def test_load_model(trained_model):
    model, scaler = trained_model
    assert model is not None
    assert scaler is not None

def test_get_prediction():
    data = [[0.038075906, 0.05068012, 0.061696206, 0.021872354, -0.044223498, -0.03482076, -0.043400846, -0.002592261, 0.01990749, -0.017646125]]
    predictions = get_prediction(data)
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert "prediction" in predictions[0]
    assert "probabilities" in predictions[0]
