import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Attendance": 85.5,
    "Midterm_Score": 78.0,
    "Final_Score": 90.5
}

response = requests.post(url, json=data)
print("Prediction:", response.json())