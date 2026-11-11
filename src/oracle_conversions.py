
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from linear_combination_of_unitaries import *
from oracle_objects.oracle_classes import *
from phase_estimation import phase_estimation_circuit


def function_oracle_to_phase_oracle_circuit(U:FunctionOracle, epsilon=0.001) -> QuantumCircuit:
    """
    Accepts a FunctionOracle and tolerance epsilon.
    Returns a unitary O that is within epsilon of any PhaseOracle for the same function.
    """
    #Construct Grover Operator.
    working_register = QuantumRegister(U.num_working_qubits)
    aux_register = QuantumRegister(U.num_aux_qubits)
    Grover = QuantumCircuit(working_register,aux_register, name="Grover")
    flip1 =  QuantumCircuit(aux_register) #negates the phase of states where all auxilary qubits are 0.
    flip1.x([a for a in aux_register])
    flip1.mcp(math.pi, aux_register[:-1], aux_register[-1])
    flip1.x([a for a in aux_register])

    flip2 = QuantumCircuit(aux_register)
    flip2.p(math.pi,aux_register[-1])
    U_circuit = U.circuit()
    Grover.append(U_circuit.to_gate(),qargs = Grover.qubits)
    Grover.append(flip2.to_gate(), qargs = aux_register)
    Grover.append(U_circuit.inverse().to_gate(), qargs = Grover.qubits)
    Grover.append(flip1.to_gate(), qargs = aux_register)
    Grover.global_phase = math.pi #ChatGPT helped me realize that this global phase is missing from the paper.

    epsilon_prime = epsilon / 10
    M = math.ceil(2*math.log(1/epsilon_prime)/(math.log(math.log(1/epsilon_prime))))
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
        *V.qregs[0][U.num_working_qubits:],
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
def function_oracle_to_phase_oracle(U:FunctionOracle, epsilon = 0.001) -> PhaseOracle:
    """
    Uses function_oracle_to_phase_oracle_circuit and returns a PhaseOracle for the function of U.
    """
    circuit_for_phase_oracle = function_oracle_to_phase_oracle_circuit(U,epsilon = epsilon)
    class ConcretePhaseOracle(PhaseOracle):
        def __init__(self,U):
            num_working_qubits = U.num_working_qubits
            num_aux_qubits = circuit_for_phase_oracle.num_qubits - U.num_working_qubits
            f = U.f
            super().__init__(num_working_qubits,num_aux_qubits,f)

        def circuit(self):
            self.usage_count+=1
            return circuit_for_phase_oracle
    return ConcretePhaseOracle(U)

def probability_oracle_to_function_oracle_circuit(U:ProbabilityOracle) -> QuantumCircuit:
    """
    Input: U is a probability oracle (i.e. a circuit, run on the zero state
    Returns: The circuit for the FunctionOracle for the function f(x)=p(x), 
            where p is the probability of outcome x when measuring U|0> in the computational basis.
    """
    working_qubits = QuantumRegister(U.num_working_qubits)
    aux_qubits = QuantumRegister(U.num_aux_qubits)

    marker_qubit = QuantumRegister(1)
    qc = QuantumCircuit(working_qubits,aux_qubits, marker_qubit)
    U_circuit = U.circuit()
    qc.append(U_circuit.inverse().to_gate(), qargs = [*working_qubits, *aux_qubits])
    qc.x([*working_qubits,*aux_qubits])
    qc.mcx(working_qubits,marker_qubit)
    return qc

def probability_oracle_to_function_oracle(U:ProbabilityOracle) -> FunctionOracle:
    """
    Input: A probability oracle U.
    Returns: The FunctionOracle for the function f(x)=p(x), 
            where p is the probability of outcome x when measuring U|0> in the computational basis.
    """
    circuit = probability_oracle_to_function_oracle_circuit(U)
    return FunctionOracleFromCircuitAndFunction(U.num_working_qubits,U.num_aux_qubits+1,lambda: circuit, U.f)

def phase_oracle_to_binary_oracle(O: PhaseOracle,accuracy = 0.01, error_prob=0.01) -> BinaryOracle:
    num_eval_qubits = math.ceil(-np.log2(accuracy))+1
    num_aux_qubits = math.ceil(np.log2(2+1/(2*error_prob)))-1
    total_num_qubits = num_eval_qubits + num_aux_qubits +O.num_working_qubits + O.num_aux_qubits

    circuit = QuantumCircuit(total_num_qubits)

    if O.num_aux_qubits>0:
        for qubit_number in range(total_num_qubits-1, O.num_working_qubits+O.num_aux_qubits-1,-1):
            circuit.swap(qubit_number, qubit_number - O.num_aux_qubits)
    
    circuit.append(phase_estimation_circuit(num_eval_qubits+num_aux_qubits,O.circuit()).to_gate(), qargs = circuit.qubits)
    if O.num_aux_qubits>0:
        for qubit_number in range(O.num_working_qubits+O.num_aux_qubits, total_num_qubits):
            circuit.swap(qubit_number, qubit_number - O.num_aux_qubits)

    return BinaryOracleFromCircuit(O.num_working_qubits,num_eval_qubits,num_aux_qubits+O.num_aux_qubits,circuit)

def compose_binary_oracles(B1: BinaryOracle,B2: BinaryOracle) -> BinaryOracle:
    """
    Input: B1 and B2 are binary oracles, such that B2.num_working_qubits = B1.num_output_qubits.
    Returns: The composition oracle that performs B1, then B2.
    Note: The composition oracle maintains that the input (working) qubits are on top, followed by output qubits, followed by auxillary qubits.
    """
    assert B1.num_output_qubits == B2.num_working_qubits
    def composite_function(x):
        return B2.f(B1.f(x))
    def composite_circuit():
        initial_working_qubits = QuantumRegister(B1.num_working_qubits)
        initial_output_qubits = QuantumRegister(B1.num_output_qubits)
        initial_aux_qubits = QuantumRegister(B1.num_aux_qubits)
        final_output_qubits = QuantumRegister(B2.num_output_qubits)
        final_aux_qubits = QuantumRegister(B2.num_aux_qubits)
        qc = QuantumCircuit(initial_working_qubits,
                            final_output_qubits,
                            initial_output_qubits,
                            initial_aux_qubits,
                            final_aux_qubits,
                            )
        qc.append(B1.circuit(),qargs = [*initial_working_qubits, *initial_output_qubits, *initial_aux_qubits])
        qc.append(B2.circuit(),qargs = [*initial_output_qubits[::-1],*final_output_qubits, *final_aux_qubits])
        return qc
    to_return = BinaryOracleFromCircuit(B1.num_working_qubits,
                                        B2.num_output_qubits, 
                                        B1.num_output_qubits +B1.num_aux_qubits+B2.num_aux_qubits,
                                        composite_circuit()
                                        )
    to_return.f = composite_function
    return to_return

def marking_binary_oracle_to_phase_circuit(B: BinaryOracle) -> QuantumCircuit:
    """
    Input: B is a binary oracle, and we assume that it has only 1 output qubit.
    Returns: A circuit that negates the phase of the inputs that the binary oracle marks.
        The output circuit should return auxillary qubits to the 0 state.
    Note: The output is not a Phase Oracle, because the phase flips by (-1), but our phase oracles change the phase by a max of e^i.
    """
    assert B.num_output_qubits==1
    working_qubits = QuantumRegister(B.num_working_qubits)
    output_qubit = QuantumRegister(B.num_output_qubits)
    aux_qubits = QuantumRegister(B.num_aux_qubits)
    qc = QuantumCircuit(working_qubits,output_qubit,aux_qubits)
    qc.append(B.circuit().to_instruction(label="Binary mark"),qargs = [*working_qubits,*output_qubit,*aux_qubits])
    qc.z(output_qubit) #flip phase
    qc.append(B.circuit().inverse().to_instruction(label="Binary mark_dg"), qargs = [*working_qubits,*output_qubit,*aux_qubits]) #uncompute
    return qc

#U = random_unitary(4, seed=42)
