from math import radians
import numpy as np
from utils.blender_utils import setCamera


# Define alternative camera paths here
def cube8corners(b_empty, i):
    if i == 0:
        b_empty.rotation_euler[2] = radians(45)
        b_empty.rotation_euler[0] = radians(45)
    elif i == 1:
        b_empty.rotation_euler[2] = radians(135)
        b_empty.rotation_euler[0] = radians(45)
    elif i == 2:
        b_empty.rotation_euler[2] = radians(225)
        b_empty.rotation_euler[0] = radians(45)
    elif i == 3:
        b_empty.rotation_euler[2] = radians(315)
        b_empty.rotation_euler[0] = radians(45)
    elif i == 4:
        b_empty.rotation_euler[2] = radians(45)
        b_empty.rotation_euler[0] = radians(-45)
    elif i == 5:
        b_empty.rotation_euler[2] = radians(135)
        b_empty.rotation_euler[0] = radians(-45)
    elif i == 6:
        b_empty.rotation_euler[2] = radians(225)
        b_empty.rotation_euler[0] = radians(-45)
    elif i == 7:
        b_empty.rotation_euler[2] = radians(315)
        b_empty.rotation_euler[0] = radians(-45)
    else:
        NotImplementedError

############### Runner functions ################

# To move to the next camera
def next_cam(config, b_empty, i, cam_settings, stepsize, head_offset=None):
    if config == 'spherical':
        # Sample a random camera from a sphere
        rot = np.random.uniform(0, 1, size=3) * (1,-2*np.pi,2*np.pi)
        rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
        b_empty.rotation_euler = rot
    elif config == 'circular':
        # Rotate camera around the object in a circle
        b_empty.rotation_euler[2] += radians(stepsize)
    elif config == 'cube8corners':
        cube8corners(b_empty, i)
    elif config == '3dcustom' or config == 'multi':
        # Apply your own configuration here by calling the necessary variable from cam_settings
        b_empty.rotation_euler[2] += radians(stepsize)
    else:
        # Invalid config 
        raise NotImplementedError(f"Invalid camera config: {config}")
