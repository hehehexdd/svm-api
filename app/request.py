from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    data: List[List[float]] = Field(...)
