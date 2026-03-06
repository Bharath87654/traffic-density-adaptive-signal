from fastapi import APIRouter

router = APIRouter()

@router.get("/api/traffic-analytics")
def analytics():

    return {
        "volume":[
            {"time":"08:00","lane1":45,"lane2":30,"lane3":20,"lane4":15},
            {"time":"09:00","lane1":80,"lane2":50,"lane3":35,"lane4":25},
            {"time":"10:00","lane1":65,"lane2":40,"lane3":30,"lane4":20}
        ],

        "waitTimes":[
            {"name":"Lane 1","wait":24},
            {"name":"Lane 2","wait":35},
            {"name":"Lane 3","wait":18},
            {"name":"Lane 4","wait":12}
        ],

        "emergency":[
            {"name":"Ambulance","value":12},
            {"name":"Firetruck","value":4},
            {"name":"Police","value":8}
        ],

        "passed":[
            {"cycle":"C-1","count":120},
            {"cycle":"C-2","count":95},
            {"cycle":"C-3","count":140}
        ]
    }