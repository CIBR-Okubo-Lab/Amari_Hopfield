# script for generate images to memorize for the Amari-Hopfield network
# Tatsuo Okubo
# 2024/12/14

import numpy as np
import os
from skimage.draw import circle_perimeter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

size = (16, 16)  # size of the image

# create patterns
def create_checkerboard(size):
    return np.indices(size).sum(axis=0) % 2

def create_circle(size, center, radius):
    im_circle = np.zeros(size, dtype=np.uint8)
    rr, cc = circle_perimeter(center[0], center[1], radius, shape=size)
    im_circle[rr, cc] = 1
    return im_circle

def create_cross(size):
    # assert that size is 16, 16
    assert size == (16, 16), "Size must be (16, 16) for the plus pattern."
    cross = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    return cross

im_checkerboard = create_checkerboard(size)
im_circle = create_circle(size, center=(8, 8), radius=6)
im_cross = create_cross(size)

np.savez("data/images.npz", checkerboard=im_checkerboard, circle=im_circle, cross=im_cross)
print("Images saved to data/images.npz")