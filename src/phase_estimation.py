"""
This file implements the phase estimation circuit.
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PermutationGate, QFTGate


def phase_estimation_circuit(num_evaluation_qubits, unitary, name="QPE"):
    """
    Input: num_evaluation_qubits is the number of binary digits of the phase to calculate.
           unitary is a circuit whose phase we want to estimate.
    Output:
        A circuit:
            The input to the circuit will be initially 0 on the evaluation qubits and an eigenstate of the unitary on the working qubits.
            The circuit calculates the phase on the working qubits.
    """
    working_qubits = QuantumRegister(unitary.num_qubits)
    evaluation_qubits = QuantumRegister(num_evaluation_qubits)
    qc = QuantumCircuit(working_qubits,evaluation_qubits)
    qc.h(evaluation_qubits)
    for index,eval_qubit in enumerate(evaluation_qubits):
        qc.append(unitary.repeat(2**index).to_gate().control(1), qargs = [eval_qubit, *working_qubits])
    qc.append(QFTGate(len(evaluation_qubits)).inverse(),qargs = evaluation_qubits)
    qc.append(PermutationGate(range(len(evaluation_qubits))[::-1]), qargs=evaluation_qubits)
    return qc
    