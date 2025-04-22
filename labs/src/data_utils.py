import pandas as pd

def load_cleaned_data(path="labs/Students_Grading_Dataset.csv"):
    df = pd.read_csv(path)

    # Drop personally identifying columns
    df = df.drop(columns=["Student_ID", "First_Name", "Last_Name", "Email"], errors="ignore")

    # Encode Grade (target)
    grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    df["Grade"] = df["Grade"].map(grade_map)

    # One-hot encode categorical columns
    categorical_cols = [
        "Gender", "Department", "Extracurricular_Activities",
        "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level"
    ]
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

    # Drop any rows with missing values
    df = df.dropna()

    X = df.drop("Grade", axis=1)
    y = df["Grade"]

    return X, y
