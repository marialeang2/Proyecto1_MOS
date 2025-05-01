import requests

url = "http://localhost:8000/calculate_route"


data = {
    "points": [
        {"latitude": 52.517037, "longitude": 13.388860},
        {"latitude": 52.529407, "longitude": 13.397634}
    ]
}

"""
data = {
    "points": [
        {"latitude": 4.7110, "longitude": -74.0721},
        {"latitude": 4.6050, "longitude": -74.0835}
    ]
}
"""
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Distancia:", result["distance_meters"], "metros")
    print("Duración:", result["duration_seconds"], "segundos")
else:
    print("Error:", response.text)
