import numpy as np


RADIANS_TO_DEGREES = np.float64(57.2957763671875)
DEGREES_TO_RADIANS = np.float64(0.017453292519943295)
SCALE = np.float32(10430.378)
COS_OFFSET = np.float32(16384)

_i = np.arange(65536, dtype=np.float64)
SIN = np.sin(_i * np.pi * 2.0 / 65536.0).astype(np.float32, copy=False)


def sin(v):
    idx = np.asarray(v, dtype=np.float32) * SCALE
    return np.take(SIN, idx.astype(np.int32), mode="wrap")


def cos(v):
    idx = np.asarray(v, dtype=np.float32) * SCALE + COS_OFFSET
    return np.take(SIN, idx.astype(np.int32), mode="wrap")


def wrap_degrees(deg):
    d = np.asarray(deg, dtype=np.float32)
    return np.remainder(d + 180.0, 360.0) - 180.0


def rotate_yaw_vector(vel, old_yaw, new_yaw):
    yaw_delta = np.asarray(old_yaw - new_yaw, dtype=np.float32)
    rad = yaw_delta * DEGREES_TO_RADIANS

    c = cos(rad)
    s = sin(rad)

    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    return r @ vel
