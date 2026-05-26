import numpy as np
from qiskit.synthesis import gridsynth_rz

from gate_syntesis import gate_sequence_from_circuit


def ensure_part2_results(n_values=range(51), part2_epsilon=1e-10, part2_circuits=None, part2_sequences=None):
    circuits  = dict(part2_circuits  or {})
    sequences = dict(part2_sequences or {})
    for n in n_values:
        if n not in circuits or n not in sequences:
            circuits[n]  = gridsynth_rz(np.pi / 2**n, epsilon=part2_epsilon)
            sequences[n] = gate_sequence_from_circuit(circuits[n])
    return circuits, sequences
