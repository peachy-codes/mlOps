from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

# Load model from local path
model_uri = "./mlruns/530372547640256128/297891d16a0d4665afa8e2d8a05c49b6/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)

class InputData(BaseModel):
    Attendance: float
    Midterm_Score: float
    Final_Score: float

@app.post("/predict")
def predict(data: InputData):
    input_df = [[data.Attendance, data.Midterm_Score, data.Final_Score]]
    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}