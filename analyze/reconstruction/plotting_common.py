import numpy as np

INTENSITY_CMAP = "gray"
DIFFERENCE_CMAP = "coolwarm"
MAGNITUDE_CMAP = "viridis"
PHASE_CMAP = "twilight"
INTERPOLATION = "nearest"
PHASE_LIMITS = (-np.pi, np.pi)
PHASE_TICKS = (-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi)
PHASE_TICK_LABELS = (r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$")


def intensity_limits(*images):
    """Shared display range for comparable reconstructed intensity images."""
    upper = 1e-12
    for image in images:
        values = np.asarray(image)
        values = values[np.isfinite(values)]
        if values.size:
            upper = max(upper, float(np.percentile(np.maximum(values, 0), 99)))
    return 0.0, upper


def difference_limits(*differences):
    """Keep the sign of differences with a symmetric range around zero."""
    upper = 1e-12
    for difference in differences:
        values = np.asarray(difference)
        values = values[np.isfinite(values)]
        if values.size:
            upper = max(upper, float(np.max(np.abs(values))))
    return -upper, upper


def magnitude_limits(magnitude):
    """Zero-based range for the displayed log10(1 + |F|) magnitude."""
    values = np.asarray(magnitude)
    values = values[np.isfinite(values)]
    upper = max(float(values.max()), 1e-12) if values.size else 1e-12
    return 0.0, upper
