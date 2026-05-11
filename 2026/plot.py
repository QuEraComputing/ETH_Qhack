import os
import numpy as np
import matplotlib.pyplot as plt

from gate_syntesis import Rz_gate, Rz_gate_injected
from gate_syntesis_helpers import gate_summary, unitary_from_gate_sequence, Rz, _rz_metrics

EPSILON     = 1e-4
N_VALUES    = range(8)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def collect(n_values, epsilon):
    rows = []
    for n in n_values:
        rz     = Rz_gate(n, epsilon)
        rz_inj = Rz_gate_injected(n, epsilon)
        s      = gate_summary(rz)
        s_inj  = gate_summary(rz_inj)
        rows.append({
            "n":            n,
            "clifford":     s["clifford"],
            "clifford_inj": s_inj["clifford"],
            "CNOT_inj":     s_inj["CNOT"],
        })
    return rows


def plot_gate_counts(rows, save_path=None):
    ns            = [r["n"]            for r in rows]
    clifford      = [r["clifford"]     for r in rows]
    clifford_inj  = [r["clifford_inj"] for r in rows]
    CNOT_inj      = [r["CNOT_inj"]     for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # --- Clifford ---
    ax = axes[0]
    ax.plot(ns, clifford,     marker="o", label="Rz",            color="steelblue")
    ax.plot(ns, clifford_inj, marker="s", label="Rz (injected)", color="darkorange", linestyle="--")
    ax.set_title("Clifford gates")
    ax.set_xlabel("n")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(ns)

    # --- CNOT (injected only; Rz has 0) ---
    ax = axes[1]
    ax.plot(ns, CNOT_inj, marker="s", color="darkorange")
    ax.set_title("CNOT gates  (Rz injected)")
    ax.set_xlabel("n")
    ax.grid(True)
    ax.set_xticks(ns)

    fig.suptitle(f"Gate counts vs n  (ε={EPSILON})", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()





def statevector_fidelity(n_values, epsilon, psi0=None):
    if psi0 is None:
        psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>

    rows = []
    for n in n_values:
        gate  = Rz_gate(n, epsilon)
        m     = _rz_metrics.get(id(gate))
        seq   = m.sequence
        theta = m.theta

        psi_approx = unitary_from_gate_sequence(seq) @ psi0
        psi_exact  = Rz(theta) @ psi0

        braket   = np.vdot(psi_exact, psi_approx)
        fidelity = abs(braket) ** 2
        rows.append({"n": n, "braket": braket, "fidelity": fidelity})
    return rows


def plot_fidelity(rows, save_path=None):
    ns       = [r["n"]        for r in rows]
    fidelity = [r["fidelity"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, fidelity, marker="o", color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("|⟨exact|approx⟩|²")
    ax.set_title(f"Statevector fidelity  Rz(π/2^n)  (ε={EPSILON})")
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("Computing gate counts from IR…")
    rows = collect(N_VALUES, EPSILON)

    print(f"{'n':>2}  {'Clif':>6}  {'Clif_inj':>8}  {'CNOT_inj':>8}")
    for r in rows:
        print(f"{r['n']:>2}  {r['clifford']:>6}  {r['clifford_inj']:>8}  {r['CNOT_inj']:>8}")

    plot_gate_counts(rows, save_path=os.path.join(RESULTS_DIR, "gate_counts_rz_vs_injected.png"))

    print("\nComputing statevector fidelities…")
    fid_rows = statevector_fidelity(N_VALUES, EPSILON)
    print(f"{'n':>2}  {'|<exact|approx>|²':>18}  {'braket':>30}")
    for r in fid_rows:
        print(f"{r['n']:>2}  {r['fidelity']:>18.10f}  {r['braket']:>30}")
    plot_fidelity(fid_rows, save_path=os.path.join(RESULTS_DIR, "fidelity_rz.png"))
