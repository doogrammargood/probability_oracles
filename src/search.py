from oracle_conversions import *
from qiskit import QuantumCircuit, QuantumRegister
import math
import numpy as np


def grover_operator(initial_unitary, phase_marking_unitary) -> QuantumCircuit:
    """
    Input: initial_unitary and phase_marking unitary are quantum circuits.
    Returns the grover operator, the operator Q from equation (1) here: https://arxiv.org/pdf/quant-ph/0005055
                                where initial_unitary is A and phase_marking_unitary is S_chi
    """
    working_register = QuantumRegister(initial_unitary.num_qubits)
    aux_register = QuantumRegister(phase_marking_unitary.num_qubits - initial_unitary.num_qubits)

    S0 = QuantumCircuit(working_register,name="S0")
    S0.x(working_register)
    S0.mcp(np.pi,working_register[:-1], working_register[-1])
    S0.x(working_register)
    qc = QuantumCircuit(working_register,aux_register, name="G")
    qc.append(phase_marking_unitary.to_instruction(label="V"), qargs = [*working_register,*aux_register])
    qc.append(initial_unitary.inverse().to_instruction(label="U_dg"), qargs=working_register)
    qc.append(S0.to_instruction(label="S0"), qargs = working_register)
    qc.append(initial_unitary.to_instruction(label="U"), qargs = working_register)
    qc.global_phase = np.pi
    return qc

def success_prob_to_num_iterations(prob_success) -> int:
    """
    Input: an initial probability of success.
    Output: the number of times that the Grover operator should be applied.
    Note: See Theorem 2 of https://arxiv.org/pdf/quant-ph/0005055
    """
    theta_a = np.arcsin(prob_success ** 0.5)
    return int(math.floor(np.pi/(4*theta_a)))

def grover_search_circuit(initial_unitary, marking_unitary,prob_success,m=None):
    """
    Input: two unitaries: initial_unitary prepares the state. marking_unitary flips the phase of the 'marked' elements and an initial probability of success
    Output: A quantum circuit that performs Grover search.
    """
    G = grover_operator(initial_unitary,marking_unitary)
    qc= QuantumCircuit(G.num_qubits,name="Grover")
    if m is None:
        m=success_prob_to_num_iterations(prob_success)
    qc.append(G.repeat(m).to_instruction(label="G"),qargs = qc.qubits)
    return qc

def grover_search_without_prob_success(initial_unitary, marking_unitary,max_grover_iterations = None,prob_failure =0.1,count_grover_calls=False) -> str:
    """
    input: initial_unitary and marking unitary are as in grover_search_circuit
        max_grover_iterations is the maximum number of grover iterations before giving up.
        prob_failure is an upper bound on the desired probability of failure, used to set max_grover_iterations.
        count_grover_calls is a flag that indicates whether to return the number of oracle calls or not.
    output: an outcome that is marked, or None if the search fails. If count_grover_calls is True, then we return it as a second output.
    We follow the Algorithm in Theorem 3: https://arxiv.org/pdf/quant-ph/0005055
    See Section 4 here for the choice of c: https://arxiv.org/pdf/quant-ph/9605034
    """

    #step 1
    l=0
    c=6/5
    num_grover_iterations =0

    if max_grover_iterations is None:
        if prob_failure>0:
            max_grover_iterations = 1/prob_failure * 9/4 * ( 2**initial_unitary.num_qubits )**0.5
        else:
            max_grover_iterations = math.inf

    
    while num_grover_iterations <= max_grover_iterations:
        #step 2 
        l+=1
        M = math.ceil(c**l)

        #step 3 
        initial_state = Statevector.from_int(0, dims = 2**marking_unitary.num_qubits)
        output_state = initial_state.evolve(initial_unitary)
        outcome, _ = output_state.measure(range(initial_unitary.num_qubits))
        outcome = outcome[::-1]
        outcome_vector = Statevector.from_int(0,dims = 2**(marking_unitary.num_qubits-initial_unitary.num_qubits)).tensor(Statevector.from_label(outcome))
        if np.allclose(outcome_vector, -outcome_vector.evolve(marking_unitary)):
            if not count_grover_calls:
                return outcome
            else:
                return outcome, num_grover_iterations
        
        #step 4
        initial_state = Statevector.from_int(0,dims = 2**marking_unitary.num_qubits)
        register = QuantumRegister(marking_unitary.num_qubits)
        qc = QuantumCircuit(register)
        qc.append(initial_unitary,qargs = qc.qubits[:initial_unitary.num_qubits])

        #step 5
        j=np.random.randint(1,M+1)
        num_grover_iterations+=j

        #step 6
        Q = grover_operator(initial_unitary,marking_unitary)
        qc.append(Q.repeat(j),qargs = qc.qubits[:Q.num_qubits])

        #step 7
        output_state = initial_state.evolve(qc)
        outcome, _ = output_state.measure(range(initial_unitary.num_qubits))
        outcome_vector = Statevector.from_int(0,dims = 2**(marking_unitary.num_qubits-initial_unitary.num_qubits)).tensor(Statevector.from_label(outcome))
        if np.allclose(outcome_vector, -outcome_vector.evolve(marking_unitary)):
            if not count_grover_calls:
                return outcome
            else:
                return outcome, num_grover_iterations
def dur_hoyer_search(initial_unitary, B: BinaryOracle, maximize=True,prob_failure = 0.1) -> str:
    """
    input: An initial unitary, that prepares a state. 
           A BinaryOracle whose output we want to maximize. Its num_working_qubits should be initial_unitary.num_qubits
    Output:
           An outcome such that the value that B returns is maximized (or minimized depending on the flag)
    Source: https://arxiv.org/pdf/quant-ph/9607014
    """


    num_working_qubits = initial_unitary.num_qubits
    assert num_working_qubits == B.num_working_qubits

    search_prob_failure = prob_failure/np.log(2**num_working_qubits)


    if maximize:
        threshold = -math.inf
    else:
        threshold = math.inf


    best = f"{np.random.randint(2**num_working_qubits):0{num_working_qubits}b}"
    threshold = B.apply_oracle(int(best,2))

    while True:
        
        marking_binary_oracle = compose_binary_oracles(B,BinaryOracleComparator(B.num_output_qubits, threshold, maximize=maximize))
        phase_circuit = marking_binary_oracle_to_phase_circuit(marking_binary_oracle)

        z = grover_search_without_prob_success(initial_unitary,phase_circuit, prob_failure= search_prob_failure)
        if z is None:
            return best
        #print("z",z)
        z=z[:num_working_qubits]
        new_val = B.apply_oracle(int(z,2))
        if maximize:
            if new_val >= threshold:
                threshold = new_val
                best = z
        else:
            if new_val < threshold:
                threshold = new_val
                best = z
