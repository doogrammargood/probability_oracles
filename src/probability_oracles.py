import platform
from importlib.metadata import version
from abc import ABC, abstractmethod
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import random_unitary,Operator

import matplotlib.pyplot as plt
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from src.linear_combination_of_unitaries import linear_combination_of_unitaries_circuit
#Convention: Throughout this code, I will always keep the working qubits at the top (lower binary indices) and the auxillary qubits at the bottom (higher binary indices)


#A general oracle has a circuit, a function, and a guarantee that it checks.
class Oracle(ABC):
    def __init__(self, num_qubits,num_aux_qubits, f):
        self.num_qubits = num_qubits
        self.num_aux_qubits = num_aux_qubits
        self.usage_count = 0
        self.f = f

    @abstractmethod
    def circuit(self):
        """
        Returns a circuit that does the oracle's job
        """
        pass

    @abstractmethod
    def check_guarantee(self):
        """
        Checks that the oracle behaves as expected with respect to its function.
        """
        pass

#A Function Oracle records the value of a function f as the probability of measuring 1 in the last qubit.
class FunctionOracle(Oracle):
    def __init__(self, num_qubits,num_aux_qubits, f):
        super().__init__(num_qubits,num_aux_qubits, f)
    def check_guarantee(self):
        for x in range(2**self.num_qubits):
            x_vect = Statevector.from_int(x, dims = 2**self.num_qubits)
            aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
            initial_state = aux_vect.tensor(x_vect)
            #initial_state = x_vect.tensor(aux_vect)
            final_state = initial_state.evolve(self.circuit())
            probs = final_state.probabilities(qargs=[self.num_qubits + self.num_aux_qubits - 1])
            assert np.allclose(probs, [ 1-self.f(x), self.f(x)])
#A PhaseOracle records the value of a function f by thanging the phase of |x> by e^{if(x)}.
class PhaseOracle(Oracle):
    def __init__(self,num_qubits,num_aux_qubits, f):
        super().__init__(num_qubits,num_aux_qubits, f)

    def check_guarantee(self):
        for x in range(2**self.num_qubits):
            x_vect = Statevector.from_int(x, dims = 2**self.num_qubits)
            aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
            initial_state = aux_vect.tensor(x_vect)
            final_state = initial_state.evolve(self.circuit())
            final_vector = final_state.data
            target_final_vector = np.array([np.exp(1j*self.f(x)) if index == x else 0 for index in range(2**(self.num_qubits + self.num_aux_qubits))])
            assert np.allclose(final_vector, target_final_vector,atol=0.001)
#A probability oracle records a probability distribution as the probability distribution that results from applying the oracle and measuring. Not currently used.
class ProbabilityOracle(Oracle):
    def __init__(self,num_qubits,num_aux_qubits, f):
        super().__init__(num_qubits,num_aux_qubits, f)
    def check_guarantee(self):
        x_vect = Statevector.from_int(0, dims = 2**self.num_qubits)
        aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
        initial_state = aux_vect.tensor(x_vect)
        final_state = initial_state.evolve(self.circuit())
        probs = final_state.probabilities(qargs = range(self.num_qubits))
        desired_probs = [self.f(x) for x in range(2**self.num_qubits)]
        assert np.allclose(probs, desired_probs)
class FunctionOracleExample(FunctionOracle):
    def __init__(self):
        num_qubits = 1
        num_aux_qubits = 1
        def f(x):
            if x == 0:
                return 0
            else:
                return 0.5
        super().__init__(num_qubits,num_aux_qubits,f)
    def circuit(self):
        self.usage_count += 1
        input_register = QuantumRegister(1)
        output_register = QuantumRegister(1)
        to_return = QuantumCircuit(input_register,output_register)
        to_return.ch(0,1)
        return to_return

class PhaseOracleExample(PhaseOracle):
    def __init__(self):
        num_qubits = 1
        num_aux_qubits = 1
        def f(x):
            if x == 0:
                return 0
            else:
                return 0.5
        super().__init__(num_qubits,num_aux_qubits,f)
    def circuit(self):
        self.usage_count += 1
        input_register = QuantumRegister(1)
        output_register = QuantumRegister(1)
        to_return = QuantumCircuit(input_register,output_register)
        #to_return.cp(theta=0.5, control_qubit=0, target_qubit=1)
        to_return.p(0.5, input_register[0])
        return to_return

class ProbabilityOracleExample(ProbabilityOracle):
    def __init__(self):
        num_qubits = 3
        num_aux_qubits = 0
        def f(x):
            if x == 0:
                return 0.5
            elif x == 2**self.num_qubits-1:
                return 0.5
            else:
                return 0
        super().__init__(num_qubits,num_aux_qubits,f)
    def circuit(self):
        self.usage_count += 1
        to_return = QuantumCircuit(3)
        to_return.h(0)
        to_return.cx(0,1)
        to_return.cx(0,2)
        return to_return
class FunctionOracleFromFunction(FunctionOracle):
    def __init__(self, num_qubits, f):
        super().__init__(num_qubits,1,f)
    def circuit(self):
        """
        Loops through all computational basis states of the working register,
        and performs a controlled y-rotation by self.f(x), conditioned on the working register being in a particular state.
        """
        working_qubits = QuantumRegister(self.num_qubits)
        aux_qubits = QuantumRegister(self.num_aux_qubits)
        c = QuantumCircuit(working_qubits,aux_qubits)
        for x in range(2**self.num_qubits):
            theta = 2 * np.arcsin(np.sqrt(self.f(x)))
            zeros = [i for i in range(self.num_qubits) if ((x >> i) & 1) == 0]
            if len(zeros)>0:
                c.x([working_qubits[i] for i in zeros])
            c.mcry(theta, list(working_qubits), aux_qubits[0], None, mode="noancilla")
            if len(zeros)>0:
                c.x([working_qubits[i] for i in zeros])
        return c
class PhaseOracleFromFunction(PhaseOracle):
    def __init__(self, num_qubits, f):
        super().__init__(num_qubits,0,f)

    def circuit(self):
        working_bits = QuantumRegister(self.num_qubits)
        c = QuantumCircuit(working_bits)
        D = DiagonalGate([np.exp(1j * self.f(x))  for x in range(2**self.num_qubits)])
        D_circuit = QuantumCircuit(working_bits)

        D_circuit.append(D,working_bits)
        c.append(
        D_circuit.to_gate(label=f"Diag({",".join( [ f"{np.exp(1j * self.f(x)):.1f}"  for x in range(2**self.num_qubits)] )}" + ")"),
        working_bits
        )
        return c

