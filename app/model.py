import io
import logging
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    try:
        svm_model = load('svm_model.joblib')
        scaler = load_scaler()

        logger.info("Model loaded")

        return svm_model, scaler
    except:
        return None, None
    
def load_scaler():
    try:
        scaler = load('scaler.joblib')

        logger.info("Scaler loaded")

        return scaler
    except:
        return None, None


def train_model():
    diabetes = load_diabetes()

    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(X_train, y_train)

    accuracy = svm_classifier.score(X_test, y_test)

    logger.info(f"Model trained. Accuracy: {accuracy:.2f}")

    dump(svm_classifier, 'svm_model.joblib')
    dump(scaler, 'scaler.joblib')

    return {
        "model": svm_classifier,
        "scaler": scaler,
        "accuracy": accuracy
    }

def get_prediction(data):
    svm_model, scaler = load_model()

    validate(svm_model, scaler)

    try:
        data_array = np.array(data)
        if data_array.shape[1] != 10:
            raise ValueError("Each input data point must have 10 features.")
        data_array = scaler.transform(data_array)

        predictions = svm_model.predict(data_array)
        probabilities = svm_model.predict_proba(data_array)

        prediction_results = [{"prediction": int(pred), "probabilities": prob.tolist()} for pred, prob in zip(predictions, probabilities)]

        return prediction_results
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal server error")


async def get_prediction_csv(file):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are allowed."}

    contents = await file.read()
    dataframe = pd.read_csv(io.BytesIO(contents))
    scaler = load_scaler()

    if scaler is None:
        raise HTTPException(status_code=400, detail="Scaler not found")
    
    data_array = scaler.transform(dataframe.values)
    
    if not data_array.size:
        raise ValueError("Input data array is empty")

    return get_prediction(data_array)


def evaluate_model():
    svm_model, scaler = load_model()

    validate(svm_model, scaler)

    diabetes = load_diabetes()

    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test = scaler.transform(X_test)

    accuracy = svm_model.score(X_test, y_test)

    logger.info(f"Evaluated. Accuracy: {accuracy:.2f}")

    return {"accuracy": accuracy}

def get_metadata():
    svm_model, scaler = load_model()

    validate(svm_model, scaler)

    diabetes = datasets.load_diabetes()
    
    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)

    return {
        "num_samples": len(y),
        "num_features": X.shape[1],
        "model_type": "SVM",
        "kernel": svm_model.kernel,
        "C": svm_model.C,
        "accuracy": svm_model.score(scaler.transform(X), y)
    }

def validate(svm_model, scaler):
    if svm_model is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    if scaler is None:
        raise HTTPException(status_code=400, detail="Scaler not found")
