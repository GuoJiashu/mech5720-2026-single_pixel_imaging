import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_080248_743060\black_reference.csv"

df = pd.read_csv(CSV_PATH)

time_s = df["time_s"].to_numpy()
mean_adc = df["mean_adc"].to_numpy()
std_adc = df["std_adc"].to_numpy()

print("===== Black Reference Analysis =====")

print("Number of measurements:", len(df))
print("Duration (min):", (time_s[-1] - time_s[0]) / 60)

print("Mean Black ADC:", np.mean(mean_adc))
print("Black ADC Std over time:", np.std(mean_adc, ddof=1))
print("Min Black ADC:", np.min(mean_adc))
print("Max Black ADC:", np.max(mean_adc))
print("Black ADC Range:", np.ptp(mean_adc))

print("Mean Within-Window Std:", np.mean(std_adc))
print("Maximum Within-Window Std:", np.max(std_adc))

intervals = np.diff(time_s)

print("Mean Black Interval (s):", np.mean(intervals))
print("Black Interval Std (s):", np.std(intervals, ddof=1))

# Background Drift

plt.figure(figsize=(10, 4))

plt.plot(time_s / 60, mean_adc, "o-", markersize=3)

plt.xlabel("Time (min)")
plt.ylabel("Black Mean ADC")
plt.title("Background Drift")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("background_drift.png", dpi=200)

# ADC Noise

plt.figure(figsize=(10, 4))

plt.plot(time_s / 60, std_adc, "o-", markersize=3)

plt.xlabel("Time (min)")
plt.ylabel("ADC Standard Deviation")
plt.title("Black Measurement Noise")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("background_noise.png", dpi=200)

# Simulated Black Sampling Interval

print("\n===== Sparse Black Reference Test =====")

for factor in [2, 4]:
    indices = np.arange(0, len(df), factor)
    indices = np.unique(np.append(indices, len(df) - 1))

    # Predict omitted reference values
    predicted = np.interp(time_s, time_s[indices], mean_adc[indices])

    # Evaluate only omitted anchors
    omitted = np.ones(len(df), dtype=bool)
    omitted[indices] = False
    errors = predicted[omitted] - mean_adc[omitted]

    if errors.size == 0:
        continue
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Every {16*factor} spots: MAE={mae:.4f}, RMSE={rmse:.4f} ADC")

plt.show()
