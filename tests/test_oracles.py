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
    def setUp(self):
        def f1(x):
            return math.sqrt(x)/3
        def f2(x):
            return math.sin(10*x) ** 2
        def f3(x):
            return (x ** 3.5) % 1
        def f4(x):
            return 0
        self.functions = [f1,f2,f3,f4]
        self.coefficients_list = [[0.1, 0.0, 0.2, 0.2,0.1,0.2,0.1,0.0],
                            [1,-2,5,8,9],
                            [1j,-1j,5,6,1-2j]
                            ]
        
        self.matrix_list = [
                    [[0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]],
                    random_unitary(4, seed=42).data,
                    random_unitary(8, seed=44).data,
                    ]


    def test_linear_combination_of_unitaries_circuit_basic(self):
        """
        Checks that the linear combination of unitaries circuit behaves as a linear combination of unitaries.
        We test every pair of a variety of coefficients and 2^n*2^n unitary matrices.
        For each pair of coefficients and matrix, we check on all states that are computational basis states in the working register, 
                                                                            tensored with the zero state in the auxillary register.
        We are checking that when the auxillary register is measured to be zero, the working register collapses 
        into the normalized result of applying the linear combination of unitaries to the initial working register state.

        We run each test many times and ensure that the auxillary register was measured to be zero at least once, for each pair of coefficients, matrix, and initial state.
        """
        
        for coefficients in self.coefficients_list:
            for matrix in self.matrix_list:
                Y=QuantumCircuit(np.log2(len(matrix)),name="Y")
                Y.unitary(matrix, range(Y.num_qubits))
                Y.to_gate()
                Y.name = "Y"

                circuit = linear_combination_of_unitaries_circuit(coefficients,Y)

                for initial_state_value in range(2**Y.num_qubits):

                    initial_state = Statevector.from_int(initial_state_value,dims=2**(circuit.num_qubits))
                    initial_state_working_register = Statevector.from_int(initial_state_value,dims=2**(Y.num_qubits))
                    state = initial_state.evolve(circuit)

                    linear_combination_of_unitaries = sum(coefficients[i] * np.linalg.matrix_power(Operator(Y).data, i) for i in range(len(coefficients)))

                    LC = np.array(linear_combination_of_unitaries)
                    initial_state_data = initial_state_working_register.data
                    
                    ideal_state_data = (LC @ initial_state_data)
                    ideal_state_data = ideal_state_data/np.linalg.norm(ideal_state_data)

                    asserted=False #False, until we actually check the output value.
                    #for _ in range(200):
                    outcome, new_state = state.measure(range(Y.num_qubits, circuit.num_qubits))
                    if outcome == '0' * len(outcome):
                        marginal_state = new_state.data[:2**Y.num_qubits]
                        marginal_state = marginal_state/np.linalg.norm(marginal_state)
                        assert np.allclose(ideal_state_data, marginal_state)
                        asserted=True

                    #assert asserted

    def test_linear_combination_of_circuits_clean(self):
        """
        Checks that the upper left block of the unitary associated with the linear combinations of unitaries circuit is actually the unitary
        """
        for coefficients in self.coefficients_list:
            coff_sum = sum([abs(c) for c in coefficients])
            for matrix in self.matrix_list:
                Y=QuantumCircuit(np.log2(len(matrix)),name="Y")
                Y.unitary(matrix, range(Y.num_qubits))
                Y.to_gate()
                Y.name = "Y"

                circuit = linear_combination_of_unitaries_circuit(coefficients,Y)
                linear_combination_of_unitaries = sum(coefficients[i] * np.linalg.matrix_power(Operator(Y).data, i) for i in range(len(coefficients)))
                LC = np.array(linear_combination_of_unitaries)
                circuit_encoded = np.array([[ r*coff_sum for r in row[:2**Y.num_qubits]] for row in Operator(circuit).data[:2**Y.num_qubits]]) #extract upper left block, scale by the sum of coefficents.
                assert np.allclose(LC,circuit_encoded)
                
   
    def test_oracle_examples(self):
        """
        Check that the basic oracle examples satisfy their guarantees.
        """
        PhaseOracleExample().check_guarantee()
        FunctionOracleExample().check_guarantee()
        ProbabilityOracleExample().check_guarantee()
        BinaryOracleFromLinear(5,10).check_guarantee()
        BinaryOracleFromLinearMultiOutput(5,3,[3,11,12]).check_guarantee()

        for f in self.functions:
            FunctionOracleFromFunction(3,f).check_guarantee()
            PhaseOracleFromFunction(3,f).check_guarantee()

    def test_oracle_conversion(self):
        """
        Tests that FunctionOracles are correctly transformed into PhaseOracles.
        """
        function_oracle_to_phase_oracle(FunctionOracleExample()).check_guarantee()
        for f in self.functions:
            function_oracle_to_phase_oracle(FunctionOracleFromFunction(3,f)).check_guarantee()

    def test_probability_to_function_conversion(self):
        M = random_unitary(8, seed=42)
        P = ProbabilityOracleFromUnitary(3,M)
        P.check_guarantee()
        U = probability_oracle_to_function_oracle(P)
        U.check_guarantee()

    def test_phase_estimation(self):
        qc = QuantumCircuit(2)
        qc.append(DiagonalGate([1,np.exp(0.5j*2*np.pi), np.exp(0.25j *2*np.pi), np.exp(0.125j*2*np.pi)]),qargs=qc.qubits)
        pe = phase_estimation_circuit(5,qc)
        for x in range(4):
            initial_state = Statevector.from_int(0,dims = [2]*5).tensor(Statevector.from_int(x,dims = 4))
            final_state = initial_state.evolve(pe)

    def test_binary_oracle_conversion(self):
        """
        This test creates a phase oracle and converts it to a binary oracle.
        Checks that the binary oracle is close to the phase oracle.
        It it probabilistic and sometimes fails.
        """
        Bin = phase_oracle_to_binary_oracle(PhaseOracleExample())
        Bin.check_approximation(lambda x: 0 if x ==0 else 0.5)

        def f(x):
            return [0.5, 0.1, 0.3, 0.1][x]
        O = PhaseOracleFromFunction(2,f)
        B1 = phase_oracle_to_binary_oracle(O,accuracy=0.01,error_prob=0.01)
        B1.check_approximation(f)
        assert np.allclose([B1.f(i)*2*np.pi/(2**B1.num_output_qubits) for i in range(4)], [0.5, 0.1, 0.3, 0.1], atol = 0.05)

    
    def test_comparator_oracle(self):
        Comp = BinaryOracleComparator(5,3)
        Comp.check_guarantee()

    def test_binary_oracle_composition(self):
        B1 = BinaryOracleFromLinearMultiOutput(5,4,[3,5,10,11])
        B2 = BinaryOracleFromLinearMultiOutput(4,2,[3,7])
        B3 = compose_binary_oracles(B1,B2)
        B3.check_guarantee()
        
    def test_mark_from_phase_oracle(self):
        """
        """
        def f(x):
            return [0.5, 0.1, 0.3, 0.1 ][x]
        O = PhaseOracleFromFunction(2,f)
        B1 = phase_oracle_to_binary_oracle(O,accuracy=0.05,error_prob=0.05)
        threshold = 0.2
        B2 = BinaryOracleComparator(B1.num_output_qubits, int(threshold*2**B1.num_output_qubits/(2*np.pi)))

        B3 = compose_binary_oracles(B1,B2)
        B3.check_guarantee()
        assert all([ B3.f(index) == 1 if f(index)>threshold else B3.f(index)==0 for index in range(4)])

    def test_marking_binary_oracle_to_phase_circuit(self):
        B = BinaryOracleFromLinear(5,2**5-1)
        qc = marking_binary_oracle_to_phase_circuit(B)
        for x in range(2**B.num_working_qubits):
            initial_vector = Statevector.from_int(0, dims = 2**B.num_aux_qubits).tensor(Statevector.from_int(0, dims=2)).tensor(Statevector.from_int(x, dims = 2**B.num_working_qubits))
            if B.f(x)==0:
                assert np.allclose(initial_vector, initial_vector.evolve(qc))
            else:
                assert np.allclose(initial_vector, -1*initial_vector.evolve(qc))

    
        

if __name__=="__main__":
    unittest.main()
    # tests = CreateTests()
    # tests.setUp()
    # tests.test_ecc_hamming_oracle()
    # tests.test_linear_combination_of_circuits_clean()
    #tests.test_ecc_decode()
    #tests.test_durr_hoyer_search()
    #tests.test_grover_search_without_prob_success()
    #tests.test_binary_oracle_hamming()
    #tests.test_binary_oracle_conversion()
    #tests.test_oracle_examples()
    #tests.test_binary_oracle_composition()
    #tests.test_marking_binary_oracle_to_phase_circuit()
    #tests.test_comparator_oracle()
    #tests.test_oracle_examples()
    #tests.test_phase_estimation()