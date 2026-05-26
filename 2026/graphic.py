import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ============================================================
# Data from your printed results
# ============================================================

# n-sweep at p = 0.010
n_values = np.array([0, 1, 2, 3])

f_ideal_n = np.array([
    1.00000,
    1.00000,
    1.00000,
    1.00000,
])

f_noisy_n = np.array([
    1.00000,
    1.00000,
    0.66667,
    0.00000,
])

# noise sweep for n = 2
p_values = np.array([
    0.000,
    0.005,
    0.010,
    0.030,
    0.050,
])

f_noise = np.array([
    1.00000,
    1.00000,
    1.00000,
    0.66667,
    0.50000,
])


# ============================================================
# Plot 1: n-sweep at fixed noise p = 0.010
# ============================================================

plt.figure(figsize=(7.5, 4.5))

plt.plot(
    n_values,
    f_ideal_n,
    marker="s",
    linestyle="--",
    label="Ideal Part 4 simulation",
)

plt.plot(
    n_values,
    f_noisy_n,
    marker="o",
    linestyle="-",
    label="Noisy Part 4 simulation",
)

plt.xlabel("n")
plt.ylabel("Logical state fidelity")
plt.title("Part 4 noisy simulation: n-sweep at p = 0.010")
plt.xticks(n_values)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("results/part4_noisy_n_sweep.png", dpi=300)
plt.show()


# ============================================================
# Plot 2: noise sweep for n = 2, without variance bars
# ============================================================

plt.figure(figsize=(7.5, 4.5))

plt.plot(
    p_values,
    f_noise,
    marker="o",
    linestyle="-",
    label="Noisy simulation, n = 2",
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="Ideal fidelity",
)

plt.xlabel("Physical noise probability p")
plt.ylabel("Logical state fidelity")
plt.title("Part 4 noisy simulation: noise sweep for n = 2")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("results/part4_noise_sweep_n2_no_variance.png", dpi=300)
plt.show()


print("Saved:")
print("  results/part4_noisy_n_sweep.png")
print("  results/part4_noise_sweep_n2_no_variance.png")