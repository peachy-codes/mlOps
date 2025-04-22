from metaflow import FlowSpec, step, Parameter
from data_utils import load_cleaned_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

class StudentTrainingFlow(FlowSpec):

    data_path = Parameter("data_path", default="labs/Students_Grading_Dataset.csv")

    @step
    def start(self):
        print("Loading and splitting data...")
        self.X, self.y = load_cleaned_data(self.data_path)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.param_grid = [
            {"C": 0.01, "solver": "liblinear"},
            {"C": 0.1, "solver": "liblinear"},
            {"C": 1.0, "solver": "liblinear"},
            {"C": 10.0, "solver": "liblinear"},
            {"C": 0.1, "solver": "lbfgs"},
            {"C": 1.0, "solver": "lbfgs"},
            {"C": 10.0, "solver": "lbfgs"},
        ]
        self.next(self.train_model, foreach="param_grid")

    @step
    def train_model(self):
        import signal
        from sklearn.exceptions import ConvergenceWarning
        import warnings

        class TimeoutException(Exception): pass
        def handler(signum, frame): raise TimeoutException()

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        params = self.input
        self.hyperparams = params

        try:
            model = LogisticRegression(C=params["C"], solver=params["solver"], max_iter=1000)
            model.fit(self.X_train, self.y_train)
            acc = accuracy_score(self.y_val, model.predict(self.X_val))

            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric("val_accuracy", acc)

            self.model = model
            self.accuracy = acc
            print(f"Training these parameters: {params}  and getting this accuracy: acc={acc:.2f}")

        except TimeoutException:
            print(f"We had another incident of freezing")
            self.model = None
            self.accuracy = -1

        except Exception as e:
            print(f"These paramteres just threw an error! {params}: {e}")
            self.model = None
            self.accuracy = -1

        finally:
            signal.alarm(0)

        self.next(self.choose_best)

    @step
    def choose_best(self, inputs):
        valid_inputs = [i for i in inputs if i.accuracy != -1]
        if not valid_inputs:
            raise ValueError("Check your params girl")

        best = max(valid_inputs, key=lambda x: x.accuracy)
        self.model = best.model
        self.best_params = best.hyperparams
        self.best_acc = best.accuracy
        print(f"Optimal parameters are {self.best_params} getting this accuracy {self.best_acc:.2f}")
        self.next(self.register)

    @step
    def register(self):
        print("Registering")
        with mlflow.start_run():
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_val_accuracy", self.best_acc)
            mlflow.sklearn.log_model(self.model, "model", registered_model_name="StudentGradeModel")
        self.next(self.end)

    @step
    def end(self):
        print("Training complete. Best accuracy:", self.best_acc)

if __name__ == "__main__":
    StudentTrainingFlow()