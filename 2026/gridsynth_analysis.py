import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit.synthesis import gridsynth_rz

T_GATES = {"t", "tdg"}


def analyze_epsilon(theta: float, epsilons: list[float]) -> tuple[list[float], list[float], list[float]]:
    t_counts, clifford_counts, valid_eps = [], [], []
    for eps in epsilons:
        if theta < eps:
            continue  # rotazione minore di epsilon → identità, skip
        try:
            circ = gridsynth_rz(theta, epsilon=eps)
            ops = circ.count_ops()
            t_counts.append(sum(ops.get(g, 0) for g in T_GATES))
            clifford_counts.append(sum(v for g, v in ops.items() if g not in T_GATES))
            valid_eps.append(eps)
        except BaseException:
            pass
    return t_counts, clifford_counts, valid_eps


def fit_log(epsilons, counts):
    """Fit counts ~ a * log2(1/eps) + b. Returns (a, b, r2)."""
    x = np.log2(1.0 / np.array(epsilons))
    y = np.array(counts, dtype=float)
    if np.all(y == y[0]):
        return 0.0, float(y[0]), 1.0
    try:
        (a, b), _ = curve_fit(lambda x, a, b: a * x + b, x, y)
        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return float(a), float(b), float(r2)
    except Exception:
        return None, None, None


def plot_gate_counts(thetas: dict[str, float], epsilons: list[float]) -> None:
    """Plot T-count e Clifford-count vs log2(1/ε) per ogni angolo, con fit overlay."""
    n_plots = len(thetas)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for ax, (label, theta) in zip(axes, thetas.items()):
        t_counts, cliff_counts, valid_eps = analyze_epsilon(theta, epsilons)
        if not valid_eps:
            continue

        x_log = np.log2(1.0 / np.array(valid_eps))
        ax.plot(x_log, t_counts, "o-", label="T-count", color="tab:red")
        ax.plot(x_log, cliff_counts, "s--", label="Clifford-count", color="tab:blue")

        for counts, color, name in [(t_counts, "tab:red", "T"), (cliff_counts, "tab:blue", "Clf")]:
            a, b, r2 = fit_log(valid_eps, counts)
            if a is not None and r2 > 0.9:
                x_fine = np.linspace(x_log.min(), x_log.max(), 200)
                ax.plot(x_fine, a * x_fine + b, "--", color=color, alpha=0.4,
                        label=f"{name}: {a:.2f}·log₂(1/ε)+{b:.1f} (R²={r2:.3f})")

        ax.set_xlabel("log₂(1/ε)")
        ax.set_ylabel("Gate count")
        ax.set_title(f"θ = {label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("gridsynth_rz: gate counts vs ε  [fit: a·log₂(1/ε) + b]", fontsize=13)
    plt.tight_layout()
    plt.savefig("gridsynth_analysis.png", dpi=150)
    plt.show()
    print("Plot salvato in gridsynth_analysis.png")


def run_all(n_max: int, epsilons: list[float], n_start: int = 1):
    """Calcola i fit per n=n_start..n_max e stampa la tabella delle slope."""
    print(f"\n{'n':>4}  {'theta':>10}  {'T slope (a)':>12}  {'T intercept (b)':>16}  "
          f"{'Clf slope (a)':>14}  {'Clf intercept (b)':>18}  {'R² T':>7}  {'R² Clf':>7}")
    print("-" * 100)

    ns, t_slopes, t_intercepts, clf_slopes, clf_intercepts, t_r2s, clf_r2s = [], [], [], [], [], [], []

    for n in range(n_start, n_max + 1):
        theta = np.pi / 2**n
        t_counts, cliff_counts, valid_eps = analyze_epsilon(theta, epsilons)

        if len(valid_eps) < 3:
            print(f"{n:>4}  {'π/2^'+str(n):>10}  {'(no data)':>12}")
            continue

        t_a, t_b, t_r2 = fit_log(valid_eps, t_counts)
        c_a, c_b, c_r2 = fit_log(valid_eps, cliff_counts)

        print(f"{n:>4}  {'π/2^'+str(n):>10}  {t_a:>12.4f}  {t_b:>16.4f}  "
              f"{c_a:>14.4f}  {c_b:>18.4f}  {t_r2:>7.4f}  {c_r2:>7.4f}")

        ns.append(n)
        t_slopes.append(t_a)
        t_intercepts.append(t_b)
        clf_slopes.append(c_a)
        clf_intercepts.append(c_b)
        t_r2s.append(t_r2)
        clf_r2s.append(c_r2)

    return (np.array(ns), np.array(t_slopes), np.array(t_intercepts),
            np.array(clf_slopes), np.array(clf_intercepts),
            np.array(t_r2s), np.array(clf_r2s))


def find_max_n(epsilons: list[float], n_max: int = 200) -> list[int]:
    """Per ciascun ε, trova il massimo n tale che gridsynth_rz(π/2^n, ε) NON sia l'identità."""
    max_ns = []
    for eps in epsilons:
        last_nontrivial = 0
        for n in range(1, n_max + 1):
            theta = np.pi / 2**n
            try:
                circ = gridsynth_rz(theta, epsilon=eps)
                total = sum(circ.count_ops().values())
                if total > 0:
                    last_nontrivial = n
                else:
                    break  # da qui in avanti θ è ancora più piccolo, sempre identità
            except BaseException:
                continue
        max_ns.append(last_nontrivial)
        print(f"  ε={eps:.0e}  →  n_max = {last_nontrivial}")
    return max_ns


def plot_max_n(epsilons: list[float], max_ns: list[int]) -> None:
    """Plot del max n esprimibile come sequenza Clifford+T vs epsilon."""
    eps_arr = np.array(epsilons)
    log_inv_eps = np.log2(1.0 / eps_arr)

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(log_inv_eps, max_ns, "o-", color="tab:purple", markersize=8, label="n_max osservato")

    # confronto teorico: π/2^n < ε  ⟺  n > log2(π/ε) = log2(π) + log2(1/ε)
    theory = np.log2(np.pi) + log_inv_eps
    ax.plot(log_inv_eps, theory, "--", color="tab:orange", alpha=0.7,
            label="teoria: log₂(π) + log₂(1/ε)")

    ax.set_xlabel("log₂(1/ε)")
    ax.set_ylabel("n_max  (max n per cui Rz(π/2ⁿ) ≠ identità)")
    ax.set_title("Massimo n esprimibile come sequenza Clifford+T vs precisione ε")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # secondo asse x con epsilon
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ax2.set_xticks(log_inv_eps)
    ax2.set_xticklabels([f"{e:.0e}" for e in eps_arr], rotation=45, fontsize=8)
    ax2.set_xlabel("ε")

    plt.tight_layout()
    plt.savefig("gridsynth_max_n.png", dpi=150)
    plt.show()
    print("Plot salvato in gridsynth_max_n.png")


def plot_slopes(ns, t_slopes, t_intercepts, clf_slopes, clf_intercepts, t_r2s, clf_r2s):
    """Plot di slope (a), intercetta (b) e R² del fit  count ~ a·log₂(1/ε) + b."""
    _, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(ns, t_slopes, "o-", color="tab:red", label="T-count  a")
    ax1.plot(ns, clf_slopes, "s-", color="tab:blue", label="Clifford  a")
    ax1.axhline(3.0, color="tab:red", linestyle="--", alpha=0.5, label="theory T: a≈3")
    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.4)
    ax1.set_ylabel("Slope  a")
    ax1.set_title("Fit  count ~ a·log₂(1/ε) + b")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns, t_intercepts, "o-", color="tab:red", label="T-count  b")
    ax2.plot(ns, clf_intercepts, "s-", color="tab:blue", label="Clifford  b")
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.4)
    ax2.set_ylabel("Intercetta  b")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(ns, t_r2s, "o-", color="tab:red", label="R² T-count")
    ax3.plot(ns, clf_r2s, "s-", color="tab:blue", label="R² Clifford")
    ax3.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax3.set_xlabel("n  (θ = π/2ⁿ)")
    ax3.set_ylabel("R²")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gridsynth_slopes.png", dpi=150)
    plt.show()
    print("Plot salvato in gridsynth_slopes.png")


def plot_global_fit(n_min: int, n_max: int, epsilons: list[float]) -> tuple[float, float]:
    """Single plot: T-count data for all n on one axis + one global linear fit."""
    all_x, all_y_t, all_y_c = [], [], []

    cmap = plt.get_cmap("viridis")
    ns_range = list(range(n_min, n_max + 1))
    colors = [cmap(i / max(len(ns_range) - 1, 1)) for i in range(len(ns_range))]

    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(16, 6))

    for color, n in zip(colors, ns_range):
        theta = np.pi / 2**n
        t_counts, cliff_counts, valid_eps = analyze_epsilon(theta, epsilons)
        if not valid_eps:
            continue
        x_log = np.log2(1.0 / np.array(valid_eps))
        ax_t.scatter(x_log, t_counts, color=color, s=12, alpha=0.6)
        ax_c.scatter(x_log, cliff_counts, color=color, s=12, alpha=0.6)
        all_x.extend(x_log.tolist())
        all_y_t.extend(t_counts)
        all_y_c.extend(cliff_counts)

    all_x = np.array(all_x)
    all_y_t = np.array(all_y_t, dtype=float)
    all_y_c = np.array(all_y_c, dtype=float)

    results = {}
    for ax, all_y, tag, theory_a in [(ax_t, all_y_t, "T", 3.0), (ax_c, all_y_c, "Clifford", 4.7)]:
        (a, b), _ = curve_fit(lambda x, a, b: a * x + b, all_x, all_y)
        x_fine = np.linspace(all_x.min(), all_x.max(), 300)
        ax.plot(x_fine, a * x_fine + b, "r-", linewidth=2,
                label=f"Global fit: {a:.3f}·log₂(1/ε) + {b:.2f}")
        ax.plot(x_fine, theory_a * x_fine, "k--", linewidth=1.2, alpha=0.6,
                label=f"Theory: {theory_a}·log₂(1/ε)")
        ax.set_xlabel("log₂(1/ε)")
        ax.set_ylabel("Gate count")
        ax.set_title(f"{tag}-count — global fit  a={a:.4f}  b={b:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        results[tag] = (a, b)
        print(f"Global fit {tag}-count:  a = {a:.4f}  b = {b:.4f}")

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=n_min, vmax=n_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("n  (θ = π/2ⁿ)", rotation=270, labelpad=15)

    plt.suptitle(f"gridsynth_rz global fit  n={n_min}..{n_max}", fontsize=13)
    plt.tight_layout()
    plt.savefig("gridsynth_global_fit.png", dpi=150)
    plt.show()
    print("Plot salvato in gridsynth_global_fit.png")
    return results


if __name__ == "__main__":
    epsilons = [10 ** (-e) for e in range(3, 50)]
    N_MAX = 100

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-start", type=int, default=1,
                        help="n di partenza per entrambi i plot (default: 1)")
    parser.add_argument("--n-detail", type=int, default=10,
                        help="n finale per il plot dettagliato (default: 10)")
    parser.add_argument("--n-global-max", type=int, default=40,
                        help="n massimo per il global-fit plot (default: 40)")
    args, _ = parser.parse_known_args()

    n_start = max(1, args.n_start)
    thetas_detail = {f"π/2^{n}  (n={n})": np.pi / 2**n for n in range(n_start, args.n_detail + 1)}
    plot_gate_counts(thetas_detail, epsilons)

    ns, t_slopes, t_intercepts, clf_slopes, clf_intercepts, t_r2s, clf_r2s = run_all(N_MAX, epsilons, n_start=n_start)

    mask = ns >= max(3, n_start)
    print(f"\nMedia slope (a) T-count  (n>=3): {t_slopes[mask].mean():.4f}")
    print(f"Media slope (a) Clifford (n>=3): {clf_slopes[mask].mean():.4f}")
    print(f"Media intercetta (b) T-count  (n>=3): {t_intercepts[mask].mean():.4f}")
    print(f"Media intercetta (b) Clifford (n>=3): {clf_intercepts[mask].mean():.4f}")

    plot_slopes(ns, t_slopes, t_intercepts, clf_slopes, clf_intercepts, t_r2s, clf_r2s)

    print("\nGlobal fit su tutti gli n...")
    plot_global_fit(n_start, args.n_global_max, epsilons)

    print("\nCalcolo n_max(ε)...")
    max_ns = find_max_n(epsilons, n_max=N_MAX)
    plot_max_n(epsilons, max_ns)
