import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.restoration import denoise_tv_chambolle

# ==========================================
# Configuration
# ==========================================

NPZ_PATH = Path(
    r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_232207_754080\point_scan_data.npz"
)

UPSCALE_FACTOR = 4
TV_WEIGHT = 0.05
GAMMA = 0.85

# ==========================================
# Load original ADC response
# ==========================================

data = np.load(NPZ_PATH)
raw = data["response_adc"].astype(np.float64)

if not np.all(np.isfinite(raw)):
    raise ValueError("Incomplete or invalid ADC data")

print("Original resolution:", raw.shape)

# ==========================================
# Normalise for image processing
# ==========================================

low = float(np.min(raw))
high = float(np.max(raw))

scale = max(high - low, 1e-12)

normalised = (raw - low) / scale

# ==========================================
# Method 1: Lanczos interpolation
# ==========================================

new_size = (raw.shape[1] * UPSCALE_FACTOR, raw.shape[0] * UPSCALE_FACTOR)

lanczos = cv2.resize(
    normalised.astype(np.float32), new_size, interpolation=cv2.INTER_LANCZOS4
)

# ==========================================
# Method 2: TV + Lanczos
# ==========================================

denoised = denoise_tv_chambolle(normalised, weight=TV_WEIGHT)

enhanced = cv2.resize(
    denoised.astype(np.float32), new_size, interpolation=cv2.INTER_LANCZOS4
)

# ==========================================
# Display-only Gamma Correction
# ==========================================

enhanced = np.clip(enhanced, 0, 1)

gamma_image = enhanced**GAMMA

# ==========================================
# Comparison
# ==========================================

fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

images = [normalised, lanczos, enhanced, gamma_image]

titles = ["Original 32x32", "Lanczos 128x128", "TV + Lanczos", "TV + Lanczos + Gamma"]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig("enhancement_comparison.png", dpi=200)

# ==========================================
# Save numerical outputs
# ==========================================

np.savez_compressed(
    "enhancement_results.npz",
    original_adc=raw,
    tv_denoised_normalised=denoised,
    lanczos_normalised=lanczos,
    enhanced_normalised=enhanced,
    gamma_display=gamma_image,
)

plt.show()
