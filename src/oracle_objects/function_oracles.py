from oracle_objects.oracles import *


#A Function Oracle records the value of a function f as the probability of measuring 1 in the last qubit.
class FunctionOracle(Oracle):
    def __init__(self, num_working_qubits,num_aux_qubits, f):
        super().__init__(num_working_qubits,num_aux_qubits, f)
    def check_guarantee(self):
        for x in range(2**self.num_working_qubits):
            x_vect = Statevector.from_int(x, dims = 2**self.num_working_qubits)
            aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
            initial_state = aux_vect.tensor(x_vect)
            #initial_state = x_vect.tensor(aux_vect)
            final_state = initial_state.evolve(self.circuit())
            probs = final_state.probabilities(qargs=[self.num_working_qubits + self.num_aux_qubits - 1])
            assert probs.shape == np.shape([ 1-self.f(x), self.f(x)])
            assert np.allclose(probs, [ 1-self.f(x), self.f(x)])
class FunctionOracleExample(FunctionOracle):
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
        to_return.ch(0,1)
        return to_return
class FunctionOracleFromCircuitAndFunction(FunctionOracle):
    def __init__(self, num_working_qubits,num_aux_qubits,circuit,f):
        self.circuit = circuit
        self.f=f
        super().__init__(num_working_qubits,num_aux_qubits,f)
    def circuit(self):
        pass
class FunctionOracleFromFunction(FunctionOracle):
    def __init__(self, num_working_qubits, f):
        super().__init__(num_working_qubits,1,f)
    def circuit(self):
        """
        Loops through all computational basis states of the working register,
        and performs a controlled y-rotation by self.f(x), conditioned on the working register being in a particular state.
        """
        working_qubits = QuantumRegister(self.num_working_qubits)
        aux_qubits = QuantumRegister(self.num_aux_qubits)
        c = QuantumCircuit(working_qubits,aux_qubits)
        for x in range(2**self.num_working_qubits):
            theta = 2 * np.arcsin(np.sqrt(self.f(x)))
            zeros = [i for i in range(self.num_working_qubits) if ((x >> i) & 1) == 0]
            if len(zeros)>0:
                c.x([working_qubits[i] for i in zeros])
            c.mcry(theta, list(working_qubits), aux_qubits[0], None, mode="noancilla")
            if len(zeros)>0:
                c.x([working_qubits[i] for i in zeros])
        return c
