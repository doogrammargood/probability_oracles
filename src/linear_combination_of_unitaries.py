from qiskit.circuit.library import DiagonalGate
from qiskit import QuantumCircuit, QuantumRegister
import math

def linear_combination_of_unitaries_circuit(alphas:list[float],Y:QuantumCircuit) -> QuantumCircuit:
    """
    Input: Y is a unitary and that we wish to implement L=sum_i alpha_i Y^i.
    Output: A quantum circuit on log(len(alpha)) + Y.num_qubits qubits that implements L. No measurement occurs.

    Note: Use this circuit on |0>|psi>, then measure the first register (log(len(alpha)) qubits). 
          When the outcome is 0, the resulting state of the unmeasured qubits is L|psi>.

    Reference: Lemma 8:
    https://arxiv.org/pdf/1511.02306
    """
    num_aux_qubits = int(math.ceil(math.log(len(alphas),2)))
    aux_register = QuantumRegister(num_aux_qubits,"aux register")
    target_register = QuantumRegister(Y.num_qubits,"target register")
    total_circuit = QuantumCircuit(target_register,aux_register)

    alpha = sum(abs(a) for a in alphas) #The L_1 norm of the alphas

    #This quantum state has a probability distribution that is the normalized absolute value of the alphas.
    distribution_state = [
        math.sqrt(abs(a) / alpha) for a in alphas
    ]
    distribution_state += [0] * (2**num_aux_qubits - len(alphas))

    V_circuit = QuantumCircuit(aux_register)
    V_circuit.prepare_state(distribution_state, aux_register)

    V = V_circuit.to_gate()
    V.name = "Prep"

    total_circuit.append(V, aux_register)

    #The phases are not mentioned in the reference, which assumes that the coefficients (alphas) are positive.
    #We apply phases to the aux register, but they jump to the target register and can be thought of as phases for Y_i, or, equivalently phases for the alpha_i.
    phases = [
        a / abs(a) if abs(a) > 0 else 1
        for a in alphas
        ]
    phases += [1] * (2**num_aux_qubits - len(alphas))

    D = DiagonalGate(phases)
    D_circuit = QuantumCircuit(aux_register)

    D_circuit.append(D,aux_register)
    total_circuit.append(
        D_circuit.to_gate(label="Diag"),
        aux_register
    )

    #This applies the n-controled Y gate construction between the aux register and the target register.
    for i, _ in enumerate(aux_register):
        Yexp = Y.power(2**i).to_gate()
        total_circuit.append(Yexp.control(1),[aux_register[i], *target_register])

    
    total_circuit.append(V.inverse(), aux_register)

    return total_circuit

