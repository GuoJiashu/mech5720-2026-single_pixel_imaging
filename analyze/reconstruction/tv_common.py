import numpy as np


def gradient(x):
    """Forward differences with zero gradient at image boundaries."""
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def gradient_adjoint(gx, gy):
    result = np.zeros_like(gx)
    result[:, :-1] -= gx[:, :-1]
    result[:, 1:] += gx[:, :-1]
    result[:-1, :] -= gy[:-1, :]
    result[1:, :] += gy[:-1, :]
    return result


def total_variation(x):
    gx, gy = gradient(x)
    return float(np.sum(np.sqrt(gx**2 + gy**2)))


def vector_shrink(gx, gy, threshold):
    magnitude = np.sqrt(gx**2 + gy**2)
    scale = np.maximum(1.0 - threshold / np.maximum(magnitude, 1e-12), 0.0)
    return gx * scale, gy * scale
