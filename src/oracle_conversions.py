import platform
from importlib.metadata import version
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from src.probability_oracles import *

def function_oracle_to_phase_oracle_circuit(U:FunctionOracle, epsilon=0.001) -> QuantumCircuit:
    """
    Accepts a FunctionOracle and tolerance epsilon.
    Returns a unitary O that is within epsilon of any PhaseOracle for the same function.
    """
    #Construct Grover Operator.
    working_register = QuantumRegister(U.num_qubits)
    aux_register = QuantumRegister(U.num_aux_qubits)
    Grover = QuantumCircuit(working_register,aux_register, name="Grover")
    flip1 =  QuantumCircuit(aux_register) #negates the phase of states where all auxilary qubits are 0.
    flip1.x([a for a in aux_register])
    flip1.mcp(math.pi, aux_register[:-1], aux_register[-1])
    flip1.x([a for a in aux_register])

    flip2 = QuantumCircuit(aux_register)
    flip2.p(math.pi,aux_register[-1])

    Grover.append(U.circuit().to_gate(),qargs = Grover.qubits)
    Grover.append(flip2.to_gate(), qargs = aux_register)
    Grover.append(U.circuit().inverse().to_gate(), qargs = Grover.qubits)
    Grover.append(flip1.to_gate(), qargs = aux_register)
    Grover.global_phase = math.pi #ChatGPT helped me realize that this global phase is missing from the paper.

    epsilon_prime = epsilon / 10
    M = int(math.ceil(2*math.log(1/epsilon_prime)/(math.log(math.log(1/epsilon_prime)))))
    betas = [ sum([math.comb(2*k, k-m) * (-1)**m*(1j)**k/(math.factorial(k)* 2**(2*k)) for k in range(abs(m), M+1)]) for m in range(-M,M+1)] #just above Equation 15 in reference

    beta = sum([abs(b) for b in betas]) #The L_1 norm
    s = math.sin(math.pi/10)
    R_matrix = [[s* beta, -(1-s**2 *beta**2 )**0.5],
                [(1-s**2 *beta**2 )**0.5, s*beta] ]
    R = QuantumCircuit(1)
    R.unitary(R_matrix, R.qubits)

    V = linear_combination_of_unitaries_circuit([b/beta for b in betas], Grover)
    V.append(Grover.to_gate().power(-M), qargs = V.qregs[0])

    extra_qubit = QuantumRegister(1, "extra")
    V.add_register(extra_qubit)
    V.append(R.to_gate(label="R"),extra_qubit)
    proj1 = QuantumCircuit(*V.qregs)
    qubits_to_flip = [
        *V.qregs[0][U.num_qubits:],
        *V.qregs[1],
        *V.qregs[2],
        ]
    proj1.x(qubits_to_flip)
    proj1.mcp(math.pi,qubits_to_flip[:-1],qubits_to_flip[-1])
    proj1.x(qubits_to_flip)

    proj2 = QuantumCircuit(*V.qregs)
    qubits_to_flip_proj2 = [
        *V.qregs[1],
        *V.qregs[2],
        ]

    proj2.x(qubits_to_flip_proj2)
    proj2.mcp(math.pi,qubits_to_flip_proj2[:-1],qubits_to_flip_proj2[-1])
    proj2.x(qubits_to_flip_proj2)

    oblivious_amplitude_grover = QuantumCircuit(*V.qregs)
    oblivious_amplitude_grover.append(proj2.to_gate(label = "P2"), qargs = V.qubits)
    oblivious_amplitude_grover.append(V.inverse().to_gate(label="V_dg"), qargs = V.qubits)
    oblivious_amplitude_grover.append(proj1.to_gate(label="P1"), qargs = V.qubits)
    oblivious_amplitude_grover.append(V.to_gate(label = "V"), qargs=V.qubits)
    oblivious_amplitude_grover.name = "Oblivious Grover"


    to_return =V.compose(oblivious_amplitude_grover.repeat(2))


    return to_return
#function_oracle_to_phase_oracle_circuit(FunctionOracleExample())
def function_oracle_to_phase_oracle(U:FunctionOracle)->PhaseOracle:
    circuit_for_phase_oracle = function_oracle_to_phase_oracle_circuit(U)
    class ConcretePhaseOracle(PhaseOracle):
        def __init__(self,U):
            num_qubits = U.num_qubits
            num_aux_qubits = circuit_for_phase_oracle.num_qubits - U.num_qubits
            f = U.f
            super().__init__(num_qubits,num_aux_qubits,f)

        def circuit(self):
            self.usage_count+=1
            return circuit_for_phase_oracle
    return ConcretePhaseOracle(U)
