class LaneManager:

    def __init__(self):

        self.lanes = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

    def reset(self):

        for lane in self.lanes:
            self.lanes[lane] = 0

    def assign_lane(self, center_x, center_y, width, height):

        mid_x = width // 2
        mid_y = height // 2

        if center_x < mid_x and center_y < mid_y:
            return 1

        if center_x >= mid_x and center_y < mid_y:
            return 2

        if center_x < mid_x and center_y >= mid_y:
            return 3

        return 4