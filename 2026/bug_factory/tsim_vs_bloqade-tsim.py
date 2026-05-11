
import tsim as ts
import re
from bloqade import tsim

def logical_to_physical_angle(logical_angle_in_pi: float, num_physical_rotations: int) -> float:
    assert (
        num_physical_rotations % 2 == 1 and num_physical_rotations > 0
    ), "k must be a positive odd integer"
    sign = -1 if (num_physical_rotations + 1) % 4 == 0 else 1
    logical_angle_in_rad = logical_angle_in_pi * np.pi
    x = np.tan(logical_angle_in_rad / 2) ** (1 / num_physical_rotations)
    theta_phys = 2 * np.arctan(x)
    return float(sign * theta_phys / np.pi)


def load_star_circuit_bloqade(d: int, logical_angle_in_pi: float) -> tsim.Circuit:
    descr = """
     RX 0
    R 1
    CNOT 0 1
    DEPOLARIZE1(0.1) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    DETECTOR rec[-1]
    """
    return tsim.Circuit(descr) ## This produces error (no __init__ with str param)

def load_star_circuit_tsim(d: int, logical_angle_in_pi: float) -> ts.Circuit:
    descr = """
     RX 0
    R 1
    CNOT 0 1
    DEPOLARIZE1(0.1) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    DETECTOR rec[-1]
    """
    return ts.Circuit(descr) ## This works as intended

load_star_circuit_tsim(3, 10)
load_star_circuit_bloqade(3, 10)

