from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age:int = Field(None)
    gender: int = Field(None)
    spo2: float = Field(None)
    bmi: float = Field(None)
    bp_dia: float = Field(None)
    bp_sys: float = Field(None)
    current_step: int = Field(None)
    pulse_rate: float= Field(None)
    temperature: float = Field(None)
