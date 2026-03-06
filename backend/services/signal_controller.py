class SignalController:

    def __init__(self):

        self.min_time = 10
        self.max_time = 40

    def calculate_time(self, vehicle_count):

        if vehicle_count <= 5:
            return 10

        elif vehicle_count <= 10:
            return 20

        elif vehicle_count <= 20:
            return 30

        else:
            return 40