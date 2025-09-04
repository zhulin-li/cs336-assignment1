import math


def cosine_lr_schedule(
    t: int, T_warm_up: int, T_cosine: int, lr_max: float, lr_min: float
) -> float:
    if t <= T_warm_up:
        return lr_max * (t / T_warm_up)
    elif t >= T_cosine:
        return lr_min
    else:
        cosine = math.cos(math.pi * (t - T_warm_up) / (T_cosine - T_warm_up))
        return lr_min + (lr_max - lr_min) * (1 + cosine) / 2
