import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from wisq.qualtran_rotation_synthesis import QualtranRS

theta = np.pi / 8

qc = QuantumCircuit(1)
qc.rz(theta, 0)

pm = PassManager([QualtranRS(epsilon=1e-10)])
compiled = pm.run(qc)

print(compiled)
print(compiled.count_ops())


import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from wisq.qualtran_rotation_synthesis import QualtranRS


def synthesize_rz(theta: float, epsilon: float = 1e-10) -> QuantumCircuit:
    """Approximate Rz(theta) using a Clifford+T-like discrete gate set."""
    circuit = QuantumCircuit(1)
    circuit.rz(theta, 0)

    pass_manager = PassManager([QualtranRS(epsilon)])
    compiled = pass_manager.run(circuit)

    return compiled


def count_t_gates(circuit: QuantumCircuit) -> int:
    """Count T and Tdg gates."""
    counts = circuit.count_ops()
    return counts.get("t", 0) + counts.get("tdg", 0)


if __name__ == "__main__":
    for n in range(0, 6):
        theta = np.pi / (2 ** n)
        compiled = synthesize_rz(theta, epsilon=1e-10)

        print(f"\nn = {n}, theta = pi/{2**n}")
        print(compiled)
        print("counts:", compiled.count_ops())
        print("T-count:", count_t_gates(compiled))
        print("depth:", compiled.depth())