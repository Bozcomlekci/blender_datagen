import numpy as np

def tuple_uint8_2_tuple_float(uint8_tuple, round_off=3):
    return tuple([round(float(val)/255, round_off) for val in uint8_tuple])

def tuple_float_2_tuple_uint8(float_tuple):
    return tuple([int(val*255) for val in float_tuple])