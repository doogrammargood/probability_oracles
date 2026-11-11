"""
This file contains the abstract base class for Oracle.
Our Oracle objects inherit from this class.
The class contains two sets of qubits working and auxillary, a function f, and a circuit created by .circuit().
A class inheriting from Oracle (e.g. FunctionOracle) needs to provide a way to check that that the .circuit() represents f.
Two descendants down from Oracle, we expect a class that actually defines the method .circuit() and a function f.
"""

from abc import ABC, abstractmethod
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import math

#Convention: Throughout this code, I will always keep the working qubits at the top (lower binary indices) and the auxillary qubits at the bottom (higher binary indices)


#A general oracle has a circuit, a function, and a guarantee that it checks.
class Oracle(ABC):
    def __init__(self, num_working_qubits,num_aux_qubits, f):
        self.num_working_qubits = num_working_qubits
        self.num_aux_qubits = num_aux_qubits
        self.usage_count = 0
        self.f = f

    @abstractmethod
    def circuit(self):
        """
        Returns a circuit that does the oracle's job
        """

    @abstractmethod
    def check_guarantee(self):
        """
        Checks that the oracle behaves as expected with respect to its function.
        """
