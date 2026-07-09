import math


def distance_2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def checkpoint_hit(car_xy, checkpoint_xy, radius=0.75):
    return distance_2d(car_xy, checkpoint_xy) <= radius


def get_next_checkpoint_index(current_index, total_checkpoints):
    return (current_index + 1) % total_checkpoints