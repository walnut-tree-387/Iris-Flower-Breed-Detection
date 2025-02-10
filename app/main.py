from fastapi import FastAPI
from pydantic import BaseModel
from .model import predict_species
import torch

app = FastAPI()

# Pydantic model to validate input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
@app.get("/")
def hello() :
    return 'Hello from Iris Flower Detection Model. Please send a post request to /predict path with a sample array like this [1.0, 2.0, 3.0, 7.0] to get prediction.'

@app.post("/predict-flower")
async def predict_flower(features: IrisInput):
    feature_tensor = torch.tensor([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]], dtype=torch.float32)
    return {"predicted_flower": predict_species(feature_tensor)}
