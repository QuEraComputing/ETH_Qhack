import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Input ---
p_noise = np.array([0.000, 0.002, 0.005, 0.010, 0.020, 0.050])

# Fidelity and standard deviations (Non-Fault-Tolerant)
F_nFT = np.array([1.000000, 1.000000, 0.950000, 1.000000, 0.950000, 0.550000])
s_nFT = np.array([1.36e-07, 1.12e-07, 2.18e-01, 1.16e-07, 2.18e-01, 4.97e-01])

# Fidelity and standard deviations (Fault-Tolerant)
F_FT = np.array([1.000000, 1.000000, 1.000000, 0.947368, 0.888889, 0.875000])
s_FT = np.array([3.12e-08, 3.74e-08, 2.25e-08, 2.23e-01, 3.14e-01, 3.31e-01])

# Acceptance rate
accept_rate = np.array([1.00, 1.00, 0.95, 0.95, 0.90, 0.80])

# --- 2. Figure Creation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Performance Analysis: Fault-Tolerant vs Non-Fault-Tolerant', fontsize=14, fontweight='bold')

# --- Plot 1: Fidelity vs p_noise ---
ax1.errorbar(p_noise, F_nFT, yerr=s_nFT, fmt='-o', color='tab:red', 
             label='Non-FT', capsize=5, capthick=1.5, alpha=0.8, markersize=6)

ax1.errorbar(p_noise, F_FT, yerr=s_FT, fmt='-s', color='tab:blue', 
             label='Fault-Tolerant', capsize=5, capthick=1.5, alpha=0.8, markersize=6)

ax1.set_xlabel('Physical Error Probability ($p_{noise}$)', fontsize=11)
ax1.set_ylabel('Logical Fidelity', fontsize=11)
ax1.set_title('Fidelity Comparison', fontsize=12)
ax1.set_ylim(0.0, 1.1)
ax1.legend(loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Acceptance Rate vs p_noise ---
ax2.plot(p_noise, accept_rate * 100, '-^', color='tab:green', 
         linewidth=2, markersize=8, label='Acceptance Rate')

ax2.set_xlabel('Physical Error Probability ($p_{noise}$)', fontsize=11)
ax2.set_ylabel('Acceptance Rate (%)', fontsize=11)
ax2.set_title('Post-Selection / Accepted Syndromes', fontsize=12)
ax2.set_ylim(0, 105)
ax2.legend(loc='lower left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()