import numpy as np


def deadzone(error, threshold):
    if abs(error) < threshold:
        return 0
    return error


def hysteresis(error, u_prev, delta_on, delta_off):
    if u_prev == 0:
        if error > delta_on:
            return 1
        elif error < -delta_on:
            return -1
        else:
            return 0

    elif u_prev == 1:
        if error < -delta_on:
            return -1
        elif error < delta_off:
            return 0
        else:
            return 1

    elif u_prev == -1:
        if error > delta_on:
            return 1
        elif error > -delta_off:
            return 0
        else:
            return -1

    return 0


def saturation(u, u_min=-1, u_max=1):
    u_clip = np.clip(u, u_min, u_max)
    return u_clip
