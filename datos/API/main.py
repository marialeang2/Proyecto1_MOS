from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import httpx

#IMPORTANTE: Para correr servidor usar uvicorn main:app --reload sobre la carpeta de API.
app = FastAPI()

class Coordinate(BaseModel):
    latitude: float
    longitude: float

class RouteRequest(BaseModel):
    points: List[Coordinate]

class RouteResponse(BaseModel):
    distance_meters: float
    duration_seconds: float

OSRM_SERVER = "https://router.project-osrm.org"

@app.post("/calculate_route", response_model=RouteResponse)
async def calculate_route(request: RouteRequest):
    if len(request.points) < 2:
        return {"error": "At least two points are required"}

    # Construir el string de coordenadas
    coordinates = ";".join(f"{p.longitude},{p.latitude}" for p in request.points)

    url = f"{OSRM_SERVER}/route/v1/driving/{coordinates}?overview=false"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code != 200:
        return {"error": "Failed to get route"}

    data = response.json()
    route = data["routes"][0]

    return RouteResponse(
        distance_meters=route["distance"],
        duration_seconds=route["duration"]
    )
