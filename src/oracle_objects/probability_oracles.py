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
from phase_estimation import phase_estimation_circuit
from oracle_objects.oracles import *
#Convention: Throughout this code, I will always keep the working qubits at the top (lower binary indices) and the auxillary qubits at the bottom (higher binary indices)



#A probability oracle records a probability distribution as the probability distribution that results from applying the oracle and measuring. Not currently used.
class ProbabilityOracle(Oracle):
    def __init__(self,num_working_qubits,num_aux_qubits, f):
        super().__init__(num_working_qubits,num_aux_qubits, f)
    def check_guarantee(self):
        x_vect = Statevector.from_int(0, dims = 2**self.num_working_qubits)
        aux_vect = Statevector.from_int(0, dims = 2**self.num_aux_qubits)
        initial_state = aux_vect.tensor(x_vect)
        final_state = initial_state.evolve(self.circuit())
        probs = final_state.probabilities(qargs = range(self.num_working_qubits))
        desired_probs = [self.f(x) for x in range(2**self.num_working_qubits)]
        assert probs.shape == np.shape(desired_probs)
        assert np.allclose(probs, desired_probs)

##--EXAMPLE ORACLE CONSTRUCTIONS --##




class ProbabilityOracleExample(ProbabilityOracle):
    def __init__(self):
        num_working_qubits = 3
        num_aux_qubits = 0
        def f(x):
            if x == 0:
                return 0.5
            elif x == 2**self.num_working_qubits-1:
                return 0.5
            else:
                return 0
        super().__init__(num_working_qubits,num_aux_qubits,f)
    def circuit(self):
        self.usage_count += 1
        to_return = QuantumCircuit(3)
        to_return.h(0)
        to_return.cx(0,1)
        to_return.cx(0,2)
        return to_return
    
class ProbabilityOracleFromCircuit(ProbabilityOracle):
    def __init__(self,num_working_qubits,c):
        def circuit():
            return c
        def f(x):
            zero_state = Statevector.from_int(0,dims=2**(num_working_qubits))
            end_state = zero_state.evolve(self.circuit())
            return end_state.probabilities(qargs=range(num_working_qubits))[x]
        self.circuit = circuit
        super().__init__(num_working_qubits,0,f)
    def circuit(): #This gets defined during __init__.
        pass



class ProbabilityOracleFromUnitary(ProbabilityOracle):
    
    def __init__(self,num_working_qubits,M):
        """
        Assume M is a unitary matrix whose dimensions are 2**num_working_qubits.
        We cannot support ProbabilityOracles with aux qubits.
        """
        def circuit():
            working_qubits = QuantumRegister(self.num_working_qubits)
            qc = QuantumCircuit(working_qubits)
            qc.unitary(M, qubits = qc.qubits)
            return qc
        
        def f(x):
            zero_state = Statevector.from_int(0,dims=2**(num_working_qubits))
            end_state = zero_state.evolve(self.circuit())
            return end_state.probabilities(qargs=range(num_working_qubits))[x]
        self.circuit = circuit
        super().__init__(num_working_qubits,0,f)

    def circuit(): #This gets defined during __init__.
        pass

