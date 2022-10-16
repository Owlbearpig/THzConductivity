from imports import *


def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def get_closest_idx(arr, val):
    return np.argmin(np.abs(arr - val))


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

