from fastapi import APIRouter
from services.traffic_state import traffic_state

router = APIRouter()

@router.post("/api/signal/override")
def override():

    traffic_state.countdown = 5

    return {"message":"override triggered"}