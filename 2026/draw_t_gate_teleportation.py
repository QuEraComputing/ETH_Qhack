import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['mathtext.fontset'] = 'cm'     # Computer Modern: \rangle renders as ⟩

# figsize 16x5 → 1 unit = 1 inch in both axes, no need for equal aspect
fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])          # axes = intera figura
ax.set_facecolor('white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

OE = '#D4783A'      # orange edge
OF = '#F9DFC0'      # orange fill
BL = '#2A4FB0'      # blue (CNOT)
GR = '#555555'      # gray (classical wire / labels)
GW = GH = 0.72      # gate size (square, 0.72 in)

dy_data = 3.5
dy_anc  = 1.5
x0      = 2.4       # wire start

# ── Labels ───────────────────────────────────────────────────────────────────
ax.text(x0 - 0.18, dy_data, r'data $|\psi\rangle$',
        ha='right', va='center', fontsize=12)
ax.text(x0 - 0.18, dy_anc,  r'ancilla $|0\rangle$',
        ha='right', va='center', fontsize=12)

# ── Wires ────────────────────────────────────────────────────────────────────
Mx  = 10.8          # measurement x (needed here for ancilla wire termination)
mw, mh = 0.82, 0.72
Sx  = 13.2          # S gate x
gap = 0.13          # classical wire half-separation

ax.plot([x0, 15.5],           [dy_data, dy_data], 'k-', lw=1.6, zorder=1)
# ancilla wire stops at left edge of measurement box
ax.plot([x0, Mx - mw/2],      [dy_anc,  dy_anc],  'k-', lw=1.6, zorder=1)

ax.text(15.6, dy_data, r'$T|\psi\rangle$', ha='left', va='center', fontsize=12)

# ── Gate helper ───────────────────────────────────────────────────────────────
def gate(cx, cy, lbl, fc=OF):
    ax.add_patch(patches.FancyBboxPatch(
        (cx - GW/2, cy - GH/2), GW, GH,
        boxstyle='round,pad=0.07',
        lw=1.8, edgecolor=OE, facecolor=fc, zorder=4))
    ax.text(cx, cy, lbl, ha='center', va='center',
            fontsize=13, fontweight='bold', zorder=5)

# ── H gate ───────────────────────────────────────────────────────────────────
Hx = 4.2
gate(Hx, dy_anc, 'H')

# ── T gate ───────────────────────────────────────────────────────────────────
Tx = 5.7
gate(Tx, dy_anc, 'T')

ax.text((Hx+Tx)/2, dy_anc - 0.72,
        r'$|A\rangle = T|+\rangle$',
        ha='center', va='top', fontsize=10, color=GR)

# ── CNOT at x=8.2 ────────────────────────────────────────────────────────────
Cx = 8.2
ax.text(Cx, dy_data + 0.58, 'CNOT', ha='center', va='bottom', fontsize=11)
ax.plot([Cx, Cx], [dy_anc, dy_data], color=BL, lw=2.0, zorder=3)
ax.plot(Cx, dy_anc, 'o', color=BL, markersize=11, zorder=5)

R = 0.30
ax.add_patch(plt.Circle((Cx, dy_data), R, color='white', ec=BL, lw=2.0, zorder=4))
ax.plot([Cx-R, Cx+R], [dy_data, dy_data], color=BL, lw=2.0, zorder=5)
ax.plot([Cx, Cx],     [dy_data-R, dy_data+R], color=BL, lw=2.0, zorder=5)

# ── Measurement ──────────────────────────────────────────────────────────────
ax.add_patch(patches.FancyBboxPatch(
    (Mx - mw/2, dy_anc - mh/2), mw, mh,
    boxstyle='round,pad=0.07',
    lw=1.8, edgecolor=OE, facecolor='white', zorder=4))

# arc
th = np.linspace(np.pi, 0, 80)
ar, acy = 0.22, dy_anc - 0.06
ax.plot(Mx + ar*np.cos(th), acy + ar*0.65*np.sin(th), 'k-', lw=1.4, zorder=5)
ax.annotate('', xy=(Mx+0.19, dy_anc+0.19), xytext=(Mx, acy),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.4), zorder=6)

ax.text(Mx, dy_anc - 0.80, r'measure $M_z$',
        ha='center', va='top', fontsize=10, color=GR)

# ── Classical double wire: two continuous L-shaped rails ─────────────────────
# Each rail: horizontal at dy_anc±gap → corner → vertical up to S gate bottom
for sgn in (+1, -1):
    d = sgn * gap
    xs = [Mx + mw/2,  Sx + d,        Sx + d        ]
    ys = [dy_anc + d,  dy_anc + d,    dy_data - GH/2]
    ax.plot(xs, ys, color=GR, lw=1.6,
            solid_joinstyle='miter', solid_capstyle='butt', zorder=3)

# ── S gate ───────────────────────────────────────────────────────────────────
gate(Sx, dy_data, 'S')
ax.text(Sx, dy_data + GH/2 + 0.15, 'if $m = 1$',
        ha='center', va='bottom', fontsize=10, color=GR)

# ─────────────────────────────────────────────────────────────────────────────
plt.savefig('results/t_gate_teleportation_circuit.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("Salvato.")
