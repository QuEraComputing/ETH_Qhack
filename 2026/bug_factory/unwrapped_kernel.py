from typing import Any
from bloqade import squin, tsim
from bloqade.types import MeasurementResult, Qubit
from kirin.dialects.ilist import IList

"""
Run this multiple times. It executes correctly to me once every 2-3 executions (yes I was surprised too)
Error trace tail:
  File "/mnt/d/Repos/ETH2_test/ETH_Qhack_26_Quantum_Icing/2026/.venv/lib/python3.12/site-packages/kirin/lowering/state.py", line 101, in lower
    result = self.parent.visit(self, node)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/d/Repos/ETH2_test/ETH_Qhack_26_Quantum_Icing/2026/.venv/lib/python3.12/site-packages/kirin/lowering/python/lowering.py", line 147, in visit
    return self.registry.ast_table[name].lower(state, node)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/d/Repos/ETH2_test/ETH_Qhack_26_Quantum_Icing/2026/.venv/lib/python3.12/site-packages/kirin/lowering/python/dialect.py", line 72, in lower
    return getattr(self, f"lower_{node.__class__.__name__}", self.unreachable)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/d/Repos/ETH2_test/ETH_Qhack_26_Quantum_Icing/2026/.venv/lib/python3.12/site-packages/kirin/dialects/py/base.py", line 24, in lower_Name
    raise lowering.BuildError(f"{name} is not defined")
kirin.lowering.exception.BuildError: n is not defined

Claude says:
 inizialize_qubits is decorated with @squin.kernel at module level, so KIRIN compiles its AST immediately on import. 
 It takes z_supports: tuple as a parameter and then iterates z_supports[stab_idx]. 
 KIRIN's type inference on a bare tuple annotation is non-deterministic — it sometimes reaches into surrounding source context 
 (likely via inspect-based AST extraction that accidentally picks up q's closure variable n from the adjacent build_stabilizer_supports) to figure out the tuple's element type.

Correct execution tail:
"An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."
(I indeed did not install bloqade-tsim[cuda13])
"""

# this will help us have return types for our methods that have more intuitive names
Register = IList[Qubit, Any]
Measurement = IList[MeasurementResult, Any]

## Ignore this, it is only for making the tuples
def build_stabilizer_supports(n):
    """
    Stabilizer support for rotated surface code n x n.

    Ritorna:
        z_supports: tuple of tuples
        x_supports: tuple of tuples

    Data qubit numbering:
        0   1   2
        3   4   5
        6   7   8
    """

    if n < 3:
        raise ValueError("n deve essere almeno 3")

    if n % 2 == 0:
        raise ValueError("n deve essere dispari: 3, 5, 7, ...")

    expected = (n**2 - 1) // 2

    def q(row, col):
        return row * n + col

    z_supports = []
    x_supports = []

    # =========================
    # Z STABILIZERS
    # =========================

    # Bordo alto: coppie orizzontali
    for col in range(0, n - 1, 2):
        z_supports.append((
            q(0, col),
            q(0, col + 1),
        ))

    # Interni Z: plaquette 2x2 con parità dispari
    for row in range(n - 1):
        for col in range(n - 1):
            if (row + col) % 2 == 1:
                z_supports.append((
                    q(row, col),
                    q(row, col + 1),
                    q(row + 1, col),
                    q(row + 1, col + 1),
                ))

    # Bordo basso: coppie orizzontali sfalsate
    for col in range(1, n - 1, 2):
        z_supports.append((
            q(n - 1, col),
            q(n - 1, col + 1),
        ))

    # =========================
    # X STABILIZERS
    # =========================

    for row in range(n - 1):

        # Bordo sinistro: coppie verticali sfalsate
        if row % 2 == 1:
            x_supports.append((
                q(row, 0),
                q(row + 1, 0),
            ))

        # Interni X: plaquette 2x2 con parità pari
        for col in range(n - 1):
            if (row + col) % 2 == 0:
                x_supports.append((
                    q(row, col),
                    q(row, col + 1),
                    q(row + 1, col),
                    q(row + 1, col + 1),
                ))

        # Bordo destro: coppie verticali
        if row % 2 == 0:
            x_supports.append((
                q(row, n - 1),
                q(row + 1, n - 1),
            ))

    if len(z_supports) != expected:
        raise RuntimeError(
            f"Z stabilizers generati: {len(z_supports)}, attesi: {expected}"
        )

    if len(x_supports) != expected:
        raise RuntimeError(
            f"X stabilizers generati: {len(x_supports)}, attesi: {expected}"
        )

    return tuple(z_supports), tuple(x_supports)

@squin.kernel
def inizialize_qubits(data: Register, ancillas: Register, num_data:int, num_z:int, num_ancillas:int, z_supports:tuple) -> Measurement:
    a = squin.qalloc(3)
    squin.cx(a[0], a[1])
    for i in range(num_data):
        squin.reset(data[i])

    for i in range(num_ancillas):
        squin.reset(ancillas[i])

    for stab_idx in range(len(z_supports)):
        for data_idx in z_supports[stab_idx]:
            squin.cx(data[data_idx], ancillas[stab_idx])

    return squin.broadcast.measure(ancillas[:num_z])


def make_helpers(d, angle):
    z_supports, x_supports = build_stabilizer_supports(d)
    num_data     = d**2
    num_z        = len(z_supports)
    num_x        = len(x_supports)
    num_ancillas = num_z + num_x


    @squin.kernel  # must be a kernel, not plain fn
    def syndrome_round(data: Register, ancillas: Register) -> Measurement:
        for i in range(num_ancillas):
            squin.reset(ancillas[i])

        for i in range(num_x):
            squin.h(ancillas[num_z + i])

        for stab_idx in range(num_z):
            for data_idx in z_supports[stab_idx]:
                squin.cx(data[data_idx], ancillas[stab_idx])

        for stab_idx in range(num_x):
            for data_idx in x_supports[stab_idx]:
                squin.cx(ancillas[num_z + stab_idx], data[data_idx])

        for i in range(num_x):
            squin.h(ancillas[num_z + i])

        return squin.broadcast.measure(ancillas)

    @squin.kernel
    def physical_rotation(data:Register):
        for idx in range(0, num_data, d):
            squin.rz(angle, data[idx])


    @squin.kernel
    def cir():
        data = squin.qalloc(num_data)
        anc = squin.qalloc(num_data)
        inizialize_qubits(data, anc, num_data, num_z, num_ancillas, z_supports)

    return syndrome_round, physical_rotation, cir

_,_, cir = make_helpers(3, 1)

tsim.Circuit(cir).diagram(height=400)