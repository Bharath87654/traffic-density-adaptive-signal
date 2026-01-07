from config import settings


class SignalController:
    def get_adaptive_timing(self, current_density):

        calc_time = settings.MIN_GREEN_TIME + (current_density * settings.VEHICLE_UNIT_TIME)

        final_time = max(settings.MIN_GREEN_TIME, min(calc_time, settings.MAX_GREEN_TIME))
        return final_time