import numpy as np

INTENSITY_CMAP = "gray"
DIFFERENCE_CMAP = "coolwarm"
INTERPOLATION = "nearest"


def intensity_limits(*images):
    """Shared display range for comparable reconstructed intensity images."""
    upper = max(float(np.percentile(np.maximum(image, 0), 99)) for image in images)
    return 0.0, upper


def difference_limits(*differences):
    """Keep the sign of differences with a symmetric range around zero."""
    upper = max(float(np.max(np.abs(difference))) for difference in differences)
    return -upper, upper
