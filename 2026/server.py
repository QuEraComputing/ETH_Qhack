from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from qiskit.synthesis import gridsynth_rz

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get("APP_DIR", BASE_DIR)).resolve()
PORT = int(os.environ.get("PORT", "8018"))

app = Flask(__name__)

ALLOWED_GATES = {"h", "s", "sdg", "t", "tdg", "x", "id", "z"}
PUBLIC_GATES = {"h", "s", "sdg", "t", "tdg", "x"}


def normalize_angle_rad(theta: float) -> float:
    """Return an equivalent angle in [-pi, pi). Global phase is irrelevant on the Bloch sphere."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def gate_sequence_from_circuit(circuit) -> list[str]:
    """Extract a Clifford+T gate sequence from a one-qubit Qiskit circuit."""
    sequence: list[str] = []

    for instruction in circuit.data:
        name = instruction.operation.name.lower()

        if name in {"barrier", "delay", "id"}:
            continue

        if len(instruction.qubits) != 1:
            raise ValueError(f"Expected a one-qubit gate, found {name!r}.")

        if name == "z":
            # Z is Clifford and equals S*S, so the frontend still sees elementary gates.
            sequence.extend(["s", "s"])
            continue

        if name not in ALLOWED_GATES:
            raise ValueError(f"Unsupported gate returned by gridsynth: {name!r}.")

        sequence.append(name)

    return sequence


def rz_gridsynth_sequence(theta_rad: float, epsilon: float) -> list[str]:
    """Use Qiskit's gridsynth_rz to synthesize Rz(theta) into Clifford+T."""
    theta_rad = normalize_angle_rad(theta_rad)
    if abs(theta_rad) < 1e-15:
        return []
    circuit = gridsynth_rz(theta_rad, epsilon=float(epsilon))
    sequence = gate_sequence_from_circuit(circuit)
    bad = [gate for gate in sequence if gate not in PUBLIC_GATES]
    if bad:
        raise ValueError(f"Internal error: non-public gates found: {bad}")
    return sequence


def tagged(gates: list[str], axis: str, source: str) -> list[dict[str, str]]:
    return [{"gate": gate, "axis": axis, "source": source} for gate in gates]


def synthesize_axis(axis: str, theta_rad: float, epsilon: float) -> list[dict[str, str]]:
    """Synthesize Rx, Ry, or Rz using only Clifford wrappers plus gridsynth Rz.

    Convention: the returned list is in actual circuit execution order.
    Rx(theta) = H Rz(theta) H.
    Ry(theta) = S H Rz(theta) H Sdg as a total unitary,
    therefore execution order is Sdg, H, Rz, H, S.
    """
    axis = axis.upper()
    theta_rad = normalize_angle_rad(theta_rad)

    if abs(theta_rad) < 1e-15:
        return []

    rz_seq = rz_gridsynth_sequence(theta_rad, epsilon)

    if axis == "Z":
        return tagged(rz_seq, "Rz", "gridsynth_rz")

    if axis == "X":
        return (
            tagged(["h"], "Rx", "Clifford wrapper")
            + tagged(rz_seq, "Rx", "gridsynth_rz")
            + tagged(["h"], "Rx", "Clifford wrapper")
        )

    if axis == "Y":
        return (
            tagged(["sdg", "h"], "Ry", "Clifford wrapper")
            + tagged(rz_seq, "Ry", "gridsynth_rz")
            + tagged(["h", "s"], "Ry", "Clifford wrapper")
        )

    raise ValueError(f"Unknown axis: {axis!r}")


def validate_order(order: str) -> str:
    """Allow single-axis orders like X and Euler-style permutations like XYZ."""
    order = str(order or "XYZ").upper().strip()
    if not order:
        raise ValueError("Order cannot be empty.")
    if any(axis not in "XYZ" for axis in order):
        raise ValueError("Order may contain only X, Y, and Z.")
    if len(set(order)) != len(order):
        raise ValueError("Order cannot repeat an axis.")
    return order


def build_full_sequence(angles_deg: dict[str, Any], order: str, epsilon: float) -> list[dict[str, str]]:
    angle_map = {
        "X": np.deg2rad(float(angles_deg.get("x", 0.0))),
        "Y": np.deg2rad(float(angles_deg.get("y", 0.0))),
        "Z": np.deg2rad(float(angles_deg.get("z", 0.0))),
    }
    order = validate_order(order)

    gates: list[dict[str, str]] = []
    for axis in order:
        gates.extend(synthesize_axis(axis, angle_map[axis], epsilon))
    return gates


# --- Backend tests ---------------------------------------------------------
# These are intentionally lightweight and run at server startup.

def _c(re: float, im: float = 0.0) -> complex:
    return complex(re, im)


I = np.eye(2, dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
X = np.array([[0, 1], [1, 0]], dtype=complex)
GATE_MATS = {"h": H, "s": S, "sdg": SDG, "t": T, "tdg": TDG, "x": X}


def exact_rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


def exact_rx(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=complex,
    )


def exact_ry(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=complex,
    )


def sequence_unitary(sequence: list[str]) -> np.ndarray:
    U = I.copy()
    for gate in sequence:
        U = GATE_MATS[gate] @ U
    return U


def gate_distance(U: np.ndarray, V: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1 - min(1.0, abs(np.trace(U.conj().T @ V)) / 2))))


def run_self_tests() -> None:
    eps = 1e-8

    seq_z = [item["gate"] for item in synthesize_axis("Z", np.deg2rad(37), eps)]
    seq_x = [item["gate"] for item in synthesize_axis("X", np.deg2rad(37), eps)]
    seq_y = [item["gate"] for item in synthesize_axis("Y", np.deg2rad(37), eps)]

    assert seq_x[0] == "h" and seq_x[-1] == "h"
    assert seq_y[:2] == ["sdg", "h"] and seq_y[-2:] == ["h", "s"]
    assert all(gate in PUBLIC_GATES for gate in seq_z + seq_x + seq_y)

    theta = normalize_angle_rad(np.deg2rad(37))
    assert gate_distance(sequence_unitary(seq_z), exact_rz(theta)) < 1e-3
    assert gate_distance(sequence_unitary(seq_x), exact_rx(theta)) < 1e-3
    assert gate_distance(sequence_unitary(seq_y), exact_ry(theta)) < 1e-3

    coarse = build_full_sequence({"x": 37, "y": 0, "z": 0}, "X", 1e-4)
    fine = build_full_sequence({"x": 37, "y": 0, "z": 0}, "X", 1e-10)
    assert len(fine) >= len(coarse)

    euler = build_full_sequence({"x": 22.5, "y": 37, "z": -60}, "XYZ", 1e-8)
    axes_seen = {item["axis"] for item in euler}
    assert {"Rx", "Ry", "Rz"}.issubset(axes_seen)

    print("Backend self-tests passed.")


def _no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return _no_cache(send_from_directory(APP_DIR, "demo.html"))


@app.route("/demo.html")
def demo():
    return _no_cache(send_from_directory(APP_DIR, "demo.html"))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "qiskit.synthesis.gridsynth_rz"})


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    try:
        data = request.get_json(force=True)
        angles_deg = data.get("angles_deg", {})
        order = data.get("order", "XYZ")
        epsilon = float(data.get("epsilon", 1e-10))

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        gates = build_full_sequence(angles_deg, order, epsilon)
        flat_sequence = [item["gate"] for item in gates]
        t_count = sum(gate in {"t", "tdg"} for gate in flat_sequence)

        by_axis: dict[str, dict[str, int]] = {}
        for item in gates:
            axis = item["axis"]
            by_axis.setdefault(axis, {"length": 0, "t_count": 0})
            by_axis[axis]["length"] += 1
            by_axis[axis]["t_count"] += int(item["gate"] in {"t", "tdg"})

        return jsonify(
            {
                "gates": gates,
                "flat_sequence": flat_sequence,
                "meta": {
                    "mode": "qiskit.synthesis.gridsynth_rz",
                    "order": validate_order(order),
                    "epsilon": epsilon,
                    "sequence_length": len(flat_sequence),
                    "t_count": t_count,
                    "clifford_count": len(flat_sequence) - t_count,
                    "by_axis": by_axis,
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    run_self_tests()
    print("Server ready.")
    print(f"Open locally: http://127.0.0.1:{PORT}/demo.html")
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False, threaded=True)
