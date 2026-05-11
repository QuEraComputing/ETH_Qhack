import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ============================================================
# Results from your run
# ============================================================

labels = ["Ideal", "GeminiOneZone", "GeminiTwoZone"]

fidelity_mean = np.array([
    1.00000,
    0.66667,
    0.66667,
])

circuit_moments = np.array([
    14,     # Original noiseless circuit
    116,    # GeminiOneZone noisy circuit
    49,     # GeminiTwoZone noisy circuit
])


# ============================================================
# Plot 1: fidelity comparison, no variance bars
# ============================================================

plt.figure(figsize=(7.5, 4.5))

plt.bar(
    labels,
    fidelity_mean,
)

plt.ylabel("Logical state fidelity")
plt.title("Part 4: Steane |+_L> encoding under Gemini noise models")
plt.ylim(0.0, 1.1)
plt.grid(axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("results/part4_gemini_noise_fidelity_no_variance.png", dpi=300)
plt.show()


# ============================================================
# Plot 2: circuit moments comparison
# ============================================================

plt.figure(figsize=(7.5, 4.5))

plt.bar(
    labels,
    circuit_moments,
)

plt.ylabel("Cirq moments")
plt.title("Part 4: circuit depth after Gemini noise transformation")
plt.grid(axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("results/part4_gemini_noise_moments.png", dpi=300)
plt.show()


print("Saved:")
print("  results/part4_gemini_noise_fidelity_no_variance.png")
print("  results/part4_gemini_noise_moments.png")