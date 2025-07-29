import numpy as np
from scipy.signal import convolve2d

def apply_filter(image, kernel):
    return convolve2d(image, kernel, mode='same', boundary='symm')

def prewitt_operator(image):
    gx = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=np.float32)

    gy = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ], dtype=np.float32)

    edge_x = apply_filter(image, gx)
    edge_y = apply_filter(image, gy)
    return np.sqrt(edge_x**2 + edge_y**2)

def sobel_operator(image):
    gx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    edge_x = apply_filter(image, gx)
    edge_y = apply_filter(image, gy)
    return np.sqrt(edge_x**2 + edge_y**2)

def scharr_operator(image):
    gx = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ], dtype=np.float32)

    gy = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ], dtype=np.float32)

    edge_x = apply_filter(image, gx)
    edge_y = apply_filter(image, gy)
    return np.sqrt(edge_x**2 + edge_y**2)
