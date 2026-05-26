"""
bloch_video.py  –  Part 2: Bloch sphere animation for Rz(pi/2^n) synthesis.

Usage (from notebook or CLI):
    from bloch_video import make_bloch_video
    make_bloch_video(n=3)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_syntesis_helpers as gh
from qiskit.synthesis import gridsynth_rz


# ── helpers ──────────────────────────────────────────────────────────────────

def _state_to_bloch(psi):
    a, b = complex(psi[0]), complex(psi[1])
    x = 2.0 * float(np.real(np.conj(a) * b))
    y = 2.0 * float(np.imag(np.conj(a) * b))
    z = float(abs(a) ** 2 - abs(b) ** 2)
    return np.array([x, y, z])


def _slerp(v0, v1, t):
    """Great-circle interpolation between two points on the unit sphere."""
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-10 or n1 < 1e-10:
        return v0 * (1 - t) + v1 * t
    u0, u1 = v0 / n0, v1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if abs(dot) > 0.9999:
        r = v0 + t * (v1 - v0)
        nrm = np.linalg.norm(r)
        return r / nrm if nrm > 1e-10 else r
    omega = np.arccos(dot)
    v = (np.sin((1 - t) * omega) * u0 + np.sin(t * omega) * u1) / np.sin(omega)
    return v * (n0 * (1 - t) + n1 * t)


def _part2_rz_sequence(n: int, epsilon: float):
    if n == 0:
        return ("s", "s")
    if n == 1:
        return ("s",)
    if n == 2:
        return ("t",)
    theta = np.pi / (2 ** n)
    try:
        qc = gridsynth_rz(theta, epsilon=epsilon)
    except Exception:
        qc = gridsynth_rz(theta, epsilon=1e-4)
    return gh.gate_sequence_from_circuit(qc)


def _draw_sphere(ax):
    phi = np.linspace(0, 2 * np.pi, 80)
    th  = np.linspace(0, np.pi, 40)
    xs  = np.outer(np.cos(phi), np.sin(th))
    ys  = np.outer(np.sin(phi), np.sin(th))
    zs  = np.outer(np.ones_like(phi), np.cos(th))
    ax.plot_surface(xs, ys, zs, alpha=0.06, color="#aad4f5", linewidth=0, rcount=40, ccount=40)

    c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(c), np.sin(c), np.zeros(200), color="gray", lw=0.5, alpha=0.35)
    ax.plot(np.cos(c), np.zeros(200), np.sin(c), color="gray", lw=0.35, alpha=0.25)
    ax.plot(np.zeros(200), np.cos(c), np.sin(c), color="gray", lw=0.35, alpha=0.25)


def _draw_axes(ax, length=1.38):
    kw = dict(color="black", linewidth=0.9, arrow_length_ratio=0.08)
    for dx, dy, dz, lbl in [(length,0,0,"x"), (0,length,0,"y"), (0,0,length,"z")]:
        ax.quiver(0, 0, 0, dx, dy, dz, **kw)
        ax.text(dx, dy, dz, lbl, fontsize=11, fontweight="bold", ha="center")


def _draw_labels(ax):
    pts = {
        "|0⟩": (0, 0, 1.18),
        "|1⟩": (0, 0, -1.28),
        "|+⟩": (1.18, 0,  0),
        "|−⟩": (-1.28, 0, 0),
    }
    for name, (px, py, pz) in pts.items():
        ax.text(px, py, pz, name, fontsize=8, color="#555555", ha="center")


# ── main function ─────────────────────────────────────────────────────────────

def make_bloch_video(
    n: int = 3,
    epsilon: float = 1e-5,
    output_dir: str = "results",
    n_interp: int = 3,
    hold_frames: int = 0,
    fps: int = 30,
    trail_max: int = 300,
):
    """
    Generate a Bloch-sphere MP4 animation for the Part 2 Rz(pi/2^n) synthesis.
    Includes a side panel showing the XY cut-plane to visualise angular convergence.

    Parameters
    ----------
    n          : rotation index  (Rz(pi/2^n))
    epsilon    : gridsynth approximation tolerance
    output_dir : folder for the output MP4
    n_interp   : frames per gate transition (smoothness)
    hold_frames: extra frames to pause after each gate
    fps        : frames per second of the output video
    trail_max  : keep only the last `trail_max` Bloch points in the drawn trail
    """
    theta        = np.pi / (2 ** n)
    rz_seq       = _part2_rz_sequence(n, epsilon)
    full_seq     = ("h",) + tuple(rz_seq)
    t_count      = gh.t_count_from_sequence(rz_seq)

    gate_mat = {
        "h": gh.H, "s": gh.S, "sdg": gh.S_DAGGER,
        "t": gh.T, "tdg": gh.T_DAGGER, "x": gh.X,
    }

    # ── compute Bloch coordinates for every intermediate state ──────────────
    psi    = np.array([1.0, 0.0], dtype=complex)
    states = [psi.copy()]
    for g in full_seq:
        psi = gate_mat[g] @ psi
        states.append(psi.copy())
    bloch_pts = [_state_to_bloch(s) for s in states]

    # target: |0> → H → Rz(theta)
    psi_tgt    = gh.Rz(theta) @ (gh.H @ np.array([1.0, 0.0], dtype=complex))
    tgt_bloch  = _state_to_bloch(psi_tgt)
    tgt_phi    = np.arctan2(tgt_bloch[1], tgt_bloch[0])   # azimuthal angle of target

    # ── build frame list ─────────────────────────────────────────────────────
    frames = []
    trail  = [bloch_pts[0]]

    for i in range(len(bloch_pts) - 1):
        v0    = bloch_pts[i]
        v1    = bloch_pts[i + 1]
        label = full_seq[i].upper()

        for j in range(n_interp):
            pt = _slerp(v0, v1, j / n_interp)
            frames.append((pt, label, trail[-trail_max:]))

        trail = trail + [v1]
        for _ in range(hold_frames):
            frames.append((v1, label, trail[-trail_max:]))

    n_frames = len(frames)
    n_gates  = len(full_seq)
    print(f"n={n}  θ=π/2^{n}  sequence length={n_gates}  T-count={t_count}")
    print(f"Total animation frames: {n_frames}  ({n_frames/fps:.1f} s at {fps} fps)")

    # ── build figure: 3D sphere | XY cut-plane | XZ cut-plane ───────────────
    fig = plt.figure(figsize=(21, 7), facecolor="white")
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.set_box_aspect([1, 1, 1])
    ax2d = fig.add_subplot(132, aspect="equal")
    ax_xz = fig.add_subplot(133, aspect="equal")

    # --- 3D Bloch sphere setup ---
    _draw_sphere(ax3d)
    _draw_axes(ax3d)
    _draw_labels(ax3d)

    ax3d.scatter(*bloch_pts[0], color="steelblue", s=60, zorder=10, label="|0⟩ start")
    ax3d.scatter(*tgt_bloch, color="green", s=200, marker="*", zorder=10,
                 label=f"Target Rz(π/2^{n})")

    trail_line, = ax3d.plot([], [], [], "-", color="royalblue", alpha=0.55, linewidth=1.8)
    dot3d,      = ax3d.plot([], [], [], "o", color="red", markersize=9, zorder=11)

    ax3d.set_xlim(-1.45, 1.45); ax3d.set_ylim(-1.45, 1.45); ax3d.set_zlim(-1.45, 1.45)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left", fontsize=8, framealpha=0.8)
    title3d = ax3d.set_title("", fontsize=10, pad=10)

    # --- 2D XY cut-plane setup ---
    c = np.linspace(0, 2 * np.pi, 300)
    ax2d.plot(np.cos(c), np.sin(c), color="gray", lw=1.2, alpha=0.5)   # equator circle
    ax2d.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax2d.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax2d.set_xlim(-1.5, 1.5); ax2d.set_ylim(-1.5, 1.5)
    ax2d.set_xlabel("X (Bloch)", fontsize=10); ax2d.set_ylabel("Y (Bloch)", fontsize=10)
    ax2d.set_title("XY cut-plane  (azimuthal view)", fontsize=10)

    # axis labels
    for txt, xy in [("|+⟩", (1.3, 0)), ("|−⟩", (-1.3, 0)),
                    ("|i⟩", (0, 1.3)), ("|−i⟩", (0, -1.3))]:
        ax2d.text(*xy, txt, ha="center", va="center", fontsize=9, color="#555")

    # target line + star on cut-plane
    ax2d.plot([0, np.cos(tgt_phi)], [0, np.sin(tgt_phi)],
              "--", color="green", lw=1.2, alpha=0.7, label=f"Target φ={np.degrees(tgt_phi):.1f}°")
    ax2d.scatter(np.cos(tgt_phi), np.sin(tgt_phi),
                 color="green", s=200, marker="*", zorder=10)

    # animated elements on cut-plane
    state_line2d, = ax2d.plot([0, 1], [0, 0], "-", color="red", lw=2.0, alpha=0.85)
    dot2d,        = ax2d.plot([1], [0], "o", color="red", markersize=9, zorder=11)
    trail2d_line, = ax2d.plot([], [], "-", color="royalblue", alpha=0.4, linewidth=1.2)

    # angle arc
    arc_pts = 60
    arc_phi_arr = np.linspace(0, 0, arc_pts)
    arc_r = 0.35
    arc_line, = ax2d.plot(
        arc_r * np.cos(arc_phi_arr), arc_r * np.sin(arc_phi_arr),
        "-", color="orange", lw=2.0, alpha=0.8
    )
    angle_text = ax2d.text(0.55, 0.10, "", fontsize=10, color="darkred",
                           transform=ax2d.transAxes)
    ax2d.legend(loc="lower right", fontsize=8, framealpha=0.8)

    # --- XZ cut-plane setup (y = 0, shows polar angle θ) ---
    ax_xz.plot(np.cos(c), np.sin(c), color="gray", lw=1.2, alpha=0.5)
    ax_xz.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax_xz.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax_xz.set_xlim(-1.5, 1.5); ax_xz.set_ylim(-1.5, 1.5)
    ax_xz.set_xlabel("X (Bloch)", fontsize=10)
    ax_xz.set_ylabel("Z (Bloch)", fontsize=10)
    ax_xz.set_title("XZ cut-plane  (polar view, y=0)", fontsize=10)

    for txt, xy in [("|+⟩", (1.3, 0)), ("|−⟩", (-1.3, 0)),
                    ("|0⟩", (0, 1.3)), ("|1⟩", (0, -1.3))]:
        ax_xz.text(*xy, txt, ha="center", va="center", fontsize=9, color="#555")

    # orange disk = xz plane highlight on sphere boundary
    theta_disk = np.linspace(0, 2*np.pi, 300)
    ax_xz.fill(np.cos(theta_disk), np.sin(theta_disk),
               color="#E07B39", alpha=0.08, zorder=0)

    # target state projected onto xz (phi=0 component)
    tgt_x_xz = tgt_bloch[0]
    tgt_z_xz = tgt_bloch[2]
    ax_xz.plot([0, tgt_x_xz], [0, tgt_z_xz],
               "--", color="green", lw=1.2, alpha=0.7,
               label=f"Target θ={np.degrees(np.arctan2(tgt_x_xz, tgt_z_xz)):.1f}°")
    ax_xz.scatter([tgt_x_xz], [tgt_z_xz], color="green", s=200, marker="*", zorder=10)

    state_xz,  = ax_xz.plot([0, 1], [0, 0], "-", color="red", lw=2.0, alpha=0.85)
    dot_xz,    = ax_xz.plot([1],    [0],    "o", color="red", markersize=9, zorder=11)
    trail_xz,  = ax_xz.plot([],    [],     "-", color="royalblue", alpha=0.4, lw=1.2)

    arc_xz_pts = 60
    arc_xz_line, = ax_xz.plot([], [], "-", color="#C05000", lw=2.0, alpha=0.8)
    angle_xz_text = ax_xz.text(0.55, 0.10, "", fontsize=10, color="darkred",
                                transform=ax_xz.transAxes)
    ax_xz.legend(loc="lower right", fontsize=8, framealpha=0.8)

    # progress bar (2D axis overlay, below both subplots)
    bar_ax   = fig.add_axes([0.12, 0.03, 0.76, 0.018])
    bar_ax.set_xlim(0, n_gates); bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_ax.fill([0, n_gates, n_gates, 0], [0, 0, 1, 1], color="#dddddd", zorder=0)
    bar_fill, = bar_ax.fill([0, 0, 0, 0], [0, 0, 1, 1],
                             color="royalblue", alpha=0.7, zorder=1, label="_nolegend_")

    def _update(frame_idx):
        pt, label, trail = frames[frame_idx]

        # ── 3D update ──
        if len(trail) >= 2:
            tp = np.array(trail)
            trail_line.set_data(tp[:, 0], tp[:, 1])
            trail_line.set_3d_properties(tp[:, 2])
        dot3d.set_data([pt[0]], [pt[1]])
        dot3d.set_3d_properties([pt[2]])

        gate_idx = frame_idx // max(n_interp + hold_frames, 1)
        title3d.set_text(
            f"Part 2 – Rz(π/2^{n}) synthesis via Clifford+T\n"
            f"Gate {gate_idx + 1}/{n_gates}: {label}   "
            f"[T-count={t_count}]"
        )

        # ── 2D cut-plane update ──
        phi_cur = np.arctan2(pt[1], pt[0])
        r_xy    = np.sqrt(pt[0]**2 + pt[1]**2)   # projected radius (≤1)
        px, py  = r_xy * np.cos(phi_cur), r_xy * np.sin(phi_cur)

        # trail on cut-plane
        if len(trail) >= 2:
            tp2 = np.array(trail)
            phi_trail = np.arctan2(tp2[:, 1], tp2[:, 0])
            r_trail   = np.sqrt(tp2[:, 0]**2 + tp2[:, 1]**2)
            trail2d_line.set_data(r_trail * np.cos(phi_trail),
                                   r_trail * np.sin(phi_trail))

        state_line2d.set_data([0, px], [0, py])
        dot2d.set_data([px], [py])

        # angle arc from 0 to current phi
        arc_angles = np.linspace(0, phi_cur, arc_pts)
        arc_line.set_data(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles))

        delta_deg = np.degrees(phi_cur - tgt_phi)
        angle_text.set_text(
            f"φ = {np.degrees(phi_cur):.1f}°\n"
            f"target = {np.degrees(tgt_phi):.1f}°\n"
            f"Δ = {delta_deg:+.2f}°"
        )

        # ── XZ cut-plane update ──
        px_xz = pt[0]   # x component
        pz_xz = pt[2]   # z component

        if len(trail) >= 2:
            tp_xz = np.array(trail)
            trail_xz.set_data(tp_xz[:, 0], tp_xz[:, 2])

        state_xz.set_data([0, px_xz], [0, pz_xz])
        dot_xz.set_data([px_xz], [pz_xz])

        # polar angle arc from z-axis to state vector
        theta_cur = np.arctan2(px_xz, pz_xz)   # angle from +z axis
        arc_xz_angles = np.linspace(0, theta_cur, arc_xz_pts)
        arc_r_xz = 0.35
        arc_xz_line.set_data(arc_r_xz * np.sin(arc_xz_angles),
                              arc_r_xz * np.cos(arc_xz_angles))

        tgt_theta = np.arctan2(tgt_x_xz, tgt_z_xz)
        delta_theta = np.degrees(theta_cur - tgt_theta)
        angle_xz_text.set_text(
            f"θ = {np.degrees(theta_cur):.1f}°\n"
            f"target = {np.degrees(tgt_theta):.1f}°\n"
            f"Δ = {delta_theta:+.2f}°"
        )

        # progress bar
        frac = (gate_idx + 1) / n_gates
        bar_fill.set_xy([[0, 0], [frac * n_gates, 0],
                         [frac * n_gates, 1], [0, 1]])

        return (trail_line, dot3d, title3d, state_line2d, dot2d,
                trail2d_line, arc_line, angle_text,
                state_xz, dot_xz, trail_xz, arc_xz_line, angle_xz_text)

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=int(1000 / fps), blit=False
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bloch_sphere_part2_n{n}.mp4")
    writer   = animation.FFMpegWriter(
        fps=fps, bitrate=2500,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
    )
    ani.save(out_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    n_val = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    make_bloch_video(n=n_val)
