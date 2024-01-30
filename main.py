
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import prediction_routers

app = FastAPI()
origins = [
    "http://localhost:3000",
    "https://phc-prediction.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(prediction_routers.predictionRouter)

