from fastapi import APIRouter, status, File, UploadFile 
from app.model import train_model, get_prediction, get_prediction_csv, evaluate_model, get_metadata
from app.request import PredictRequest

router = APIRouter()

@router.post("/train/", status_code=status.HTTP_204_NO_CONTENT)
async def train():
    train_model()
    return None

@router.post("/predict/")
async def predict(predict_request: PredictRequest):
    return get_prediction(predict_request.data)

@router.post("/predict/csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    return await get_prediction_csv(file)

@router.get("/evaluate/")
async def evaluate():
    return evaluate_model()

@router.get("/metadata/")
async def metadata():
    return get_metadata()