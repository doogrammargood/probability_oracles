from oracle_conversions import *
from heavy_sampling_circuit import *
from qiskit.quantum_info import random_unitary,Operator
from qiskit.circuit.library import DiagonalGate
from phase_estimation import *
from search import *
from error_correcting_codes import *

import unittest
from unittest import TestCase


class CreateTests(TestCase):

    def test_grover_search(self):
        """
        In this example, there is 1 marked element among a search space of size 4.
        It is known that the marked element is found with probability 1 after a single application of the Grover operator.
        """
    
        qc = QuantumCircuit(2)
        qc.h([0,1])
        U = ProbabilityOracleFromCircuit(2,qc)

        #B = marking_binary_oracle_to_phase_circuit(BinaryOracleFromLinear(3,2))
        B = BinaryOracleMark(2,0)
        search_circuit = grover_search_circuit(U.circuit(),marking_binary_oracle_to_phase_circuit(B),0.25)
        zero_state = Statevector.from_int(0,dims = 2**U.circuit().num_qubits)
        initial_state = Statevector.from_int(0, dims = 2**(search_circuit.num_qubits - U.circuit().num_qubits)).tensor(zero_state.evolve(U.circuit()))
        final_state = initial_state.evolve(search_circuit)
        outcome, _ = final_state.measure([0,1])
        assert outcome == "00"

    def test_grover_search_without_prob_success(self):
        qc = QuantumCircuit(5)
        qc.h(range(5))
        initial_unitary = qc
        mark= 15
        marking_unitary = marking_binary_oracle_to_phase_circuit(BinaryOracleMark(5,15))
        outcome = grover_search_without_prob_success(initial_unitary, marking_unitary,prob_failure=0.01)
        assert outcome == f"{mark:0{len(outcome)}b}"

    
    def test_binary_oracle_hamming(self):
        H = BinaryOracleHamming(5, 0).check_guarantee()

    def test_durr_hoyer_search(self):
        B = BinaryOracleHamming(4, 0)
        initial_unitary = QuantumCircuit(4)
        initial_unitary.h(range(4))
        assert dur_hoyer_search(initial_unitary,B,maximize=False) == "0000"

    def test_ecc_hamming_oracle(self):
        E = ExampleLinearCode()
        point = E.encode_classical("0010")
        hamming_dist = BinaryOracleECCHamming(E,int(point,2))
        hamming_dist.check_guarantee()

    def test_ecc_encode(self):
        L= ExampleLinearCode()

        L2 = LinearCode(G=L.G)
        assert L2.H==L.H and L2.G==L2.G
        assert np.all(np.array(L2.G) @ np.array(L2.H).transpose() == 0) #Macwilliams and Sloane, Equation 9.
        L.generation_binary_oracle().check_guarantee()
        B=L.generation_binary_oracle()

        for i, x in enumerate([1,2,4,8]):
            assert f"{x:0{L.k}b}"[::-1] + f"{B.apply_oracle(x):0{L.n-L.k}b}" == "".join(str(abs(l)) for l in L.G[i])

        for logical_word in range(2**L.k):

            logical_vector = Statevector.from_int(logical_word,dims = 2**L.k)
            aux_vector = Statevector.from_int(0,dims = 2**(L.n-L.k))
            initial_vector = aux_vector.tensor(logical_vector)
            #checks that the circuit encodes the same encoding as the classical encoding.
            assert L.encode_classical(f"{logical_word:0{L.k}b}") == initial_vector.evolve(B.circuit()).measure()[0][::-1]

    def test_ecc_decode(self):
        L = ExampleLinearCode()

        for logical_word in range(0,2**L.k,5):
            encoded_word = L.encode_classical(f"{logical_word:0{L.k}b}")
            quantum_decode= L.decode_quantum(encoded_word)
            classical_decode = L.decode_classical(encoded_word)

            assert f"{logical_word:0{L.k}b}"== L.decode_classical(encoded_word)
            assert classical_decode==quantum_decode


            #print(f"{logical_word:0{L.k}b}", L.decode_classical(encoded_word))
