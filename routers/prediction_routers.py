
from fastapi import APIRouter, status

# To dump the models
from repo import dump_models, prediction_functions
from schemas import PredictionRequest

predictionRouter = APIRouter(
  prefix="/predictions",
  tags=['Predictions']
)
@predictionRouter.post('/', status_code=status.HTTP_200_OK)
async def getPredictions(request: PredictionRequest):
    return prediction_functions.getPredictions(request)
    
    
    # In case, we want to dump the models. We don't want to do it everytime.
    # return dump_models.getPredictions(request)
