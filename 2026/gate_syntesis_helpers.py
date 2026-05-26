from collections import namedtuple

import cirq
import numpy as np
from bloqade import squin
from bloqade.types import Qubit
from qiskit.synthesis import gridsynth_rz

_RzMeta = namedtuple("_RzMeta", ["n", "sequence", "theta", "circuit", "ancillas"])

_GATE_MAP = {
    "h":   cirq.H,
    "s":   cirq.S,
    "sdg": cirq.S**-1,
    "t":   cirq.T,
    "tdg": cirq.T**-1,
    "x":   cirq.X,
}


def _qiskit_to_cirq(qk_circ) -> cirq.Circuit:
    q = cirq.LineQubit(0)
    ops = [_GATE_MAP[instr.operation.name](q) for instr in qk_circ.data]
    return cirq.Circuit(ops)


def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


I = np.eye(2, dtype=complex)

H = 1 / np.sqrt(2) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

S = np.array([
    [1, 0],
    [0, 1j]
], dtype=complex)

T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

S_DAGGER = S.conj().T
T_DAGGER = T.conj().T


def gate_distance(U, V):
    return np.sqrt(1 - abs(np.trace(U.conj().T @ V)) / 2)


def gate_sequence_from_circuit(circuit):
    supported_gates = {"id", "h", "s", "sdg", "t", "tdg", "x"}
    sequence = []

    for instruction in circuit.data:
        name = instruction.operation.name.lower()

        if name in {"barrier", "delay"}:
            continue

        if len(instruction.qubits) != 1:
            raise ValueError(f"Expected a one-qubit circuit, found gate {name!r}.")

        if name not in supported_gates:
            raise ValueError(f"Unsupported gate from synthesized circuit: {name!r}.")

        if name != "id":
            sequence.append(name)

    return tuple(sequence)


def unitary_from_gate_sequence(sequence):
    gate_matrices = {
        "h": H,
        "s": S,
        "sdg": S_DAGGER,
        "t": T,
        "tdg": T_DAGGER,
        "x": X,
    }

    U = I.copy()
    for gate_name in sequence:
        U = gate_matrices[gate_name] @ U
    return U


def t_count_from_sequence(sequence):
    return sum(gate_name in {"t", "tdg"} for gate_name in sequence)


_rz_metrics = {}  # id(kernel) -> _RzMeta; squin Method objects don't support arbitrary attrs


def count_gates_from_kernel(kernel, verbose=False):
    from kirin.passes.inline import InlinePass
    from kirin.passes.aggressive.unroll import UnrollScf

    mt = kernel.similar()
    InlinePass(dialects=mt.dialects).fixpoint(mt)
    UnrollScf(dialects=mt.dialects).fixpoint(mt)
    InlinePass(dialects=mt.dialects).fixpoint(mt)

    gate_map = {"h": "H", "s": "S", "t": "T", "x": "X", "cx": "CX", "measure": "M"}
    counts = {}

    def walk(region):
        for block in region.blocks:
            for stmt in block.stmts:
                if verbose:
                    print(stmt.name)
                gate = gate_map.get(stmt.name)
                if gate:
                    counts[gate] = counts.get(gate, 0) + 1
                for sub in stmt.regions:
                    walk(sub)

    walk(mt.code.body)
    return counts


def print_gate_sequence(gate) -> None:
    m = _rz_metrics.get(id(gate))
    if m is None:
        raise AttributeError("Gate has no registered metrics — create it with Rz_gate().")
    seq = m.sequence
    counts = {}
    for g in seq:
        counts[g] = counts.get(g, 0) + 1
    breakdown = "  ".join(f"{g}x{n}" for g, n in counts.items())
    print(f"Rz(π/2^{m.n})  len={len(seq)}")
    print(f"  {breakdown}")
    print(f"  {'  '.join(seq)}")


def gate_summary(gate):
    c = count_gates_from_kernel(gate)
    return {
        "clifford": c.get("H", 0) + c.get("S", 0),
        "T":        c.get("T", 0),
        "CNOT":     c.get("CX", 0),
        "ancillas": c.get("M", 0),
    }


def print_metrics_split(gate) -> None:
    m = _rz_metrics.get(id(gate))
    if m is None:
        raise AttributeError("Gate has no registered metrics — create it with Rz_gate().")

    ir_counts = count_gates_from_kernel(gate)
    N = m.ancillas

    print(f"Rz(π/2^{m.n})  θ={m.theta:.6f}  ancillas={N}")
    print(f"  From IR (whole kernel, all qubits combined):")
    for g, n in sorted(ir_counts.items()):
        print(f"    {g:<6}: {n}")


# --- Postselected gate gadgets ---

@squin.kernel
def Postselected_T_gate(qubit, ancilla) -> int:
    # Same magic-state gadget as Injected_T_gate, but without feed-forward.
    # A returned 1 marks a shot that must be dropped in postselection.
    squin.h(ancilla)
    squin.t(ancilla)
    squin.cx(qubit, ancilla)
    return squin.measure(ancilla)


@squin.kernel
def Postselected_Tdg_gate(qubit, ancilla) -> int:
    measurement = Postselected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return measurement


@squin.kernel
def apply_postselected_gate_sequence(qubit, ancillas, sequence) -> int:
    ancilla_index = 0
    dropout = False

    for gate_name in sequence:
        if gate_name == "h":
            squin.h(qubit)
        elif gate_name == "s":
            squin.s(qubit)
        elif gate_name == "sdg":
            squin.s(qubit)
            squin.s(qubit)
            squin.s(qubit)
        elif gate_name == "t":
            measurement = Postselected_T_gate(qubit, ancillas[ancilla_index])
            dropout = dropout | measurement
            ancilla_index += 1
        elif gate_name == "tdg":
            measurement = Postselected_Tdg_gate(qubit, ancillas[ancilla_index])
            dropout = dropout | measurement
            ancilla_index += 1

    return dropout


def make_postselected_dropout_kernel(sequence):
    sequence = tuple(sequence)
    ancilla_count = t_count_from_sequence(sequence)

    @squin.kernel
    def postselected_dropout_kernel() -> int:
        qubits = squin.qalloc(1 + ancilla_count)
        return apply_postselected_gate_sequence(qubits[0], qubits[1:], sequence)

    return postselected_dropout_kernel


def Rz_gate_postselected(n: int, epsilon: float = 1e-10):
    """Same as Rz_gate_injected but T/Tdg gates use postselected gadgets (no feed-forward).
    Returns a kernel (qubit, ancillas) -> int where int is the dropout flag (1 = discard shot).
    For n=0,1 no ancillas are needed; the returned kernel takes only (qubit,) and returns 0."""
    if n == 0:
        @squin.kernel
        def _rz_x(qubit) -> int:
            squin.h(qubit)
            squin.s(qubit)
            squin.s(qubit)
            squin.h(qubit)
            return 0
        return _rz_x

    if n == 1:
        @squin.kernel
        def _rz_s(qubit) -> int:
            squin.s(qubit)
            return 0
        return _rz_s

    sequence = ("t",) if n == 2 else gate_sequence_from_circuit(gridsynth_rz(np.pi / 2**n, epsilon=epsilon))

    @squin.kernel
    def _rz_postselected(qubit, ancillas) -> int:
        return apply_postselected_gate_sequence(qubit, ancillas, sequence)

    return _rz_postselected


# --- Injected gate gadgets ---

@squin.kernel
def Injected_T_gate(qubit, ancilla) -> Qubit:
    # Prepare the magic state |A> = T|+> on the ancilla.
    squin.h(ancilla)
    squin.t(ancilla)

    # Entangle the data qubit with the magic-state ancilla.
    squin.cx(qubit, ancilla)

    # Measure the ancilla.
    measurement = squin.measure(ancilla)

    # Feed-forward correction.
    # If the measurement is 1, the data qubit has T^\dagger instead of T.
    # Applying S gives S T^\dagger = T.
    if measurement:
        squin.s(qubit)

    return qubit


@squin.kernel
def Injected_Tdg_gate(qubit, ancilla) -> Qubit:
    qubit = Injected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return qubit


@squin.kernel
def apply_injected_gate_sequence(qubit, ancillas, sequence) -> Qubit:
    ancilla_index = 0

    for gate_name in sequence:
        if gate_name == "h":
            squin.h(qubit)
        elif gate_name == "s":
            squin.s(qubit)
        elif gate_name == "sdg":
            squin.s(qubit)
            squin.s(qubit)
            squin.s(qubit)
        elif gate_name == "t":
            qubit = Injected_T_gate(qubit, ancillas[ancilla_index])
            ancilla_index += 1
        elif gate_name == "tdg":
            qubit = Injected_Tdg_gate(qubit, ancillas[ancilla_index])
            ancilla_index += 1

    return qubit


@squin.kernel
def Steane_measure_logical_Z_weight3(q) -> int:
    m0 = squin.measure(q[0])
    m1 = squin.measure(q[1])
    m2 = squin.measure(q[2])
    squin.measure(q[3])
    squin.measure(q[4])
    squin.measure(q[5])
    squin.measure(q[6])

    return m0 ^ m1 ^ m2

    

def statevector_fidelity(gate):
    """For each n: simulate the synthesized Rz via its gate sequence (matrix product),
    compute the exact Rz statevector, and return |<exact|approx>|^2."""
    psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>

    m      = _rz_metrics.get(id(gate))
    seq    = m.sequence
    theta  = m.theta

    psi_approx = unitary_from_gate_sequence(seq) @ psi0
    psi_exact  = Rz(theta) @ psi0

    braket    = np.vdot(psi_exact, psi_approx)           # <exact|approx>
    fidelity  = abs(braket) ** 2
    return fidelity