from oracle_objects.oracles import *
from qiskit.circuit.library import DiagonalGate



#A PhaseOracle records the value of a function f by thanging the phase of |x> by e^{if(x)}.
class PhaseOracle(Oracle):
    def __init__(self,num_working_qubits,num_aux_qubits, f):
        super().__init__(num_working_qubits,num_aux_qubits, f)

    def check_guarantee(self):
        for x in range(2**self.num_working_qubits):
            x_vect = Statevector.from_int(x, dims = 2**self.num_working_qubits)
            aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
            initial_state = aux_vect.tensor(x_vect)
            final_state = initial_state.evolve(self.circuit())
            final_vector = final_state.data
            target_final_vector = np.array([np.exp(1j*self.f(x)) if index == x else 0 for index in range(2**(self.num_working_qubits + self.num_aux_qubits))])
            assert final_vector.shape == np.shape(target_final_vector)
            assert np.allclose(final_vector, target_final_vector,atol=0.001)
class PhaseOracleExample(PhaseOracle):
    def __init__(self):
        num_working_qubits = 1
        num_aux_qubits = 1
        def f(x):
            if x == 0:
                return 0
            else:
                return 0.5
        super().__init__(num_working_qubits,num_aux_qubits,f)
    def circuit(self):
        self.usage_count += 1
        input_register = QuantumRegister(1)
        output_register = QuantumRegister(1)
        to_return = QuantumCircuit(input_register,output_register)
        #to_return.cp(theta=0.5, control_qubit=0, target_qubit=1)
        to_return.p(0.5, input_register[0])
        return to_return

class PhaseOracleFromFunction(PhaseOracle):
    def __init__(self, num_working_qubits, f):
        super().__init__(num_working_qubits,0,f)

    def circuit(self):
        working_bits = QuantumRegister(self.num_working_qubits)
        c = QuantumCircuit(working_bits)
        D = DiagonalGate([np.exp(1j * self.f(x))  for x in range(2**self.num_working_qubits)])
        D_circuit = QuantumCircuit(working_bits)

        D_circuit.append(D,working_bits)
        c.append(
        D_circuit.to_gate(label=f"Diag({",".join( [ f"{np.exp(1j * self.f(x)):.1f}"  for x in range(2**self.num_working_qubits)] )}" + ")"),
        working_bits
        )
        return c
