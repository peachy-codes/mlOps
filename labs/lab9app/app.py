from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

class request_body(BaseModel):
    reddit_comment : str

@app.on_event("startup")
def load_model():
    global model_pipeline
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")

@app.get("/")
def main():
    return {"message": "This is a model for classifying Reddit comments"}

@app.post("/predict")
def predict(data: request_body):
    X = [data.reddit_comment]
    predictions = model_pipeline.predict_proba(X)
    return {"prediction_proba_remove": float(predictions[0][1])}