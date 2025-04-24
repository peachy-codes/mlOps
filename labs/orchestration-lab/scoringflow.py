from metaflow import FlowSpec, step, Parameter, kubernetes
from data_utils import load_cleaned_data
import mlflow.sklearn
import numpy as np

from metaflow import FlowSpec, step, Parameter
from data_utils import load_cleaned_data
import mlflow.sklearn
from mlflow.tracking import MlflowClient

@kubernetes(image="python:3.10")
class ScoringFlow(FlowSpec):

    data_path = Parameter("data_path", default="Students_Grading_Dataset.csv")

    @step
    def start(self):
        self.X, self.y = load_cleaned_data(self.data_path)
        self.X_holdout = self.X.iloc[-30:]
        self.y_holdout = self.y.iloc[-30:]
        self.next(self.load_model)

    @step
    def load_model(self):
        from mlflow.tracking import MlflowClient

        print("Loading the model")
        client = MlflowClient()
        latest_versions = client.get_latest_versions("StudentGradeModel", stages=["None", "Production", "Staging"])
        latest = sorted(latest_versions, key=lambda v: int(v.version))[-1]

        # Save URI and load model
        self.model_uri = f"models:/StudentGradeModel/{latest.version}"
        self.model = mlflow.sklearn.load_model(self.model_uri)

        # Fetch and print hyperparameters from run
        run_id = latest.run_id
        run_data = client.get_run(run_id)
        self.hyperparams = run_data.data.params

        print(f"Loaded: {self.model_uri}")
        print(f"Using these hypers: {self.hyperparams}")

        self.next(self.predict)

    @step
    def predict(self):
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import datetime

        preds = self.model.predict(self.X_holdout)
        self.preds = preds.tolist()
        self.truth = self.y_holdout.tolist()
        self.accuracy = accuracy_score(self.y_holdout, preds)

        print("\n--- Predictions vs Actuals ---")
        for i, (p, t) in enumerate(zip(self.preds, self.truth)):
            print(f"Sample {i+1}: predicted = {p}, actual = {t}")
        print(f"\nHoldout accuracy: {self.accuracy:.4f}")

        # Save to CSV
        df_out = pd.DataFrame({
            "predicted": self.preds,
            "actual": self.truth
        })
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"labs/predictions_{timestamp}.csv"
        df_out.to_csv(self.output_file, index=False)
        print(f"Predictions located at: {self.output_file}")

        self.next(self.end)

    @step
    def end(self):
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"File: {self.output_file}")

if __name__ == "__main__":
    ScoringFlow()
