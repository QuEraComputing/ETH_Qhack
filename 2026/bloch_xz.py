import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(13, 10))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# ── Sphere surface ────────────────────────────────────────────────────────────
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 80)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, alpha=0.08, color='steelblue', zorder=0)

# ── Latitude/longitude lines ──────────────────────────────────────────────────
for phi in np.linspace(0, np.pi, 7):          # meridians
    xc = np.sin(phi) * np.cos(u)
    yc = np.sin(phi) * np.sin(u)
    zc = np.full_like(u, np.cos(phi))
    ax.plot(xc, yc, zc, color='steelblue', alpha=0.18, lw=0.8)

for th in np.linspace(0, 2*np.pi, 9):         # parallels
    xc = np.cos(th) * np.sin(v)
    yc = np.sin(th) * np.sin(v)
    zc = np.cos(v)
    ax.plot(xc, yc, zc, color='steelblue', alpha=0.18, lw=0.8)

# ── xz plane disk (y = 0) ────────────────────────────────────────────────────
r_disk = np.linspace(0, 1, 30)
a_disk = np.linspace(0, 2*np.pi, 120)
Rd, Ad = np.meshgrid(r_disk, a_disk)
Xd = Rd * np.cos(Ad)
Yd = np.zeros_like(Xd)
Zd = Rd * np.sin(Ad)
ax.plot_surface(Xd, Yd, Zd, alpha=0.22, color='#E07B39', zorder=1)

# ── Circle at sphere ∩ xz plane ──────────────────────────────────────────────
circ_a = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(circ_a), np.zeros_like(circ_a), np.sin(circ_a),
        color='#C05000', lw=2.2, zorder=5)

# ── Equatorial circle (z = 0) for reference ──────────────────────────────────
ax.plot(np.cos(circ_a), np.sin(circ_a), np.zeros_like(circ_a),
        color='steelblue', lw=1.0, alpha=0.5, linestyle='--')

# ── Axes arrows ───────────────────────────────────────────────────────────────
L = 1.35
for vec, col, lbl, off in [
    ([L, 0, 0], '#CC2222', 'x', (0.08, 0, 0)),
    ([0, L, 0], '#228822', 'y', (0,  0.08, 0)),
    ([0, 0, L], '#2244CC', 'z', (0, 0, 0.08)),
]:
    ax.quiver(0, 0, 0, *vec, color=col, arrow_length_ratio=0.10,
              linewidth=1.8, zorder=10)
    ax.text(vec[0]+off[0], vec[1]+off[1], vec[2]+off[2],
            lbl, color=col, fontsize=14, fontweight='bold', zorder=11)

# ── Key states ────────────────────────────────────────────────────────────────
states = {
    r'$|0\rangle$':  ( 0,    0,    1.13),
    r'$|1\rangle$':  ( 0,    0,   -1.17),
    r'$|+\rangle$':  ( 1.12, 0,    0.04),
    r'$|-\rangle$':  (-1.18, 0,    0.04),
    r'$|i\rangle$':  ( 0,    1.12, 0.04),
    r'$|-i\rangle$': ( 0,   -1.18, 0.04),
}
dot_states_xz = {r'$|0\rangle$', r'$|1\rangle$',
                 r'$|+\rangle$', r'$|-\rangle$'}

for lbl, (px, py, pz) in states.items():
    ax.scatter([px*0.91], [py*0.91], [pz*0.91],
               color='#C05000' if lbl in dot_states_xz else 'steelblue',
               s=40, zorder=12)
    ax.text(px, py, pz, lbl, fontsize=12,
            color='#C05000' if lbl in dot_states_xz else 'steelblue',
            ha='center', va='center', zorder=13)

# ── Example state in xz plane ─────────────────────────────────────────────────
theta_ex = np.pi / 3          # polar angle
state_x = np.sin(theta_ex)
state_z = np.cos(theta_ex)
ax.quiver(0, 0, 0, state_x, 0, state_z,
          color='#8B0000', linewidth=2.5, arrow_length_ratio=0.12, zorder=15)
ax.text(state_x*1.12, 0, state_z*1.12,
        r'$|\psi\rangle$', color='#8B0000', fontsize=13, zorder=16)

# ── Theta arc annotation ───────────────────────────────────────────────────────
arc_r = 0.28
arc_t = np.linspace(0, theta_ex, 60)
ax.plot(arc_r*np.sin(arc_t), np.zeros_like(arc_t), arc_r*np.cos(arc_t),
        color='#8B0000', lw=1.5)
ax.text(arc_r*0.55, 0, arc_r*0.82, r'$\theta$', color='#8B0000', fontsize=12)

# ── Layout ────────────────────────────────────────────────────────────────────
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_box_aspect([1, 1, 1])
ax.axis('off')

ax.view_init(elev=20, azim=-60)

ax.set_title('Bloch sphere — sezione sul piano $xz$  ($y=0$)',
             fontsize=13, pad=12, color='#333333')

plt.tight_layout()
plt.savefig('results/bloch_xz_plane.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("Salvato in results/bloch_xz_plane.png")
