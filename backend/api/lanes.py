from fastapi import APIRouter
from services.traffic_state import traffic_state

router = APIRouter()

@router.get("/api/lanes")
def get_lanes():

    return {"lanes":traffic_state.lanes}