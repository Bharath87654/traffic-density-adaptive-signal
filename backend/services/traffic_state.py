class TrafficState:

    def __init__(self):

        self.active_lane = 1
        self.countdown = 15

        self.lanes = [
            {"id":1,"vehicles":0,"queue":0,"signal":"red","emergency":False},
            {"id":2,"vehicles":0,"queue":0,"signal":"red","emergency":False},
            {"id":3,"vehicles":0,"queue":0,"signal":"red","emergency":False},
            {"id":4,"vehicles":0,"queue":0,"signal":"red","emergency":False}
        ]

traffic_state = TrafficState()