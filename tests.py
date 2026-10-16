from src.oracle_conversions import *
from qiskit.quantum_info import random_unitary,Operator
from qiskit.circuit.library import DiagonalGate

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
                    for _ in range(200):
                        outcome, new_state = state.measure(range(Y.num_qubits, circuit.num_qubits))
                        if outcome == '0' * len(outcome):
                            marginal_state = new_state.data[:2**Y.num_qubits]
                            marginal_state = marginal_state/np.linalg.norm(marginal_state)
                            assert np.allclose(ideal_state_data, marginal_state)
                            asserted=True
                    assert asserted

   
    def test_oracle_examples(self):
        """
        Check that the basic oracle examples satisfy their guarantees.
        """
        PhaseOracleExample().check_guarantee()
        FunctionOracleExample().check_guarantee()
        ProbabilityOracleExample().check_guarantee()

        for f in self.functions:
            FunctionOracleFromFunction(3,f).check_guarantee()
            PhaseOracleFromFunction(3,f).check_guarantee()

    def test_oracle_conversion(self):
        function_oracle_to_phase_oracle(FunctionOracleExample()).check_guarantee()
        for f in self.functions:
            function_oracle_to_phase_oracle(FunctionOracleFromFunction(3,f)).check_guarantee()
if __name__=="__main__":
    unittest.main()
    # tests = CreateTests()
    # tests.setUp()
    # tests.test_oracle_examples()