from qiskit.circuit.library import IntegerComparator, WeightedAdder

from oracle_objects.oracles import *


#A binary oracle records a function f:Z_2^n -> Z_2^d by sending |x>|0>|0> -> |x>|f(x)>|junk>.
class BinaryOracle(Oracle):
    def __init__(self,num_working_qubits, num_output_qubits, num_aux_qubits,f):
        self.num_output_qubits = num_output_qubits
        super().__init__(num_working_qubits,num_aux_qubits,f)

    def check_guarantee(self):
        for x in range(2**self.num_working_qubits):
            #print("check",x,f"{x:0{self.num_working_qubits}b}",self.apply_oracle(x), self.f(x))
            assert self.apply_oracle(x) == self.f(x)
            
    def check_approximation(self,ideal_f, epsilon=0.1):
        """
        Assume that self.f is an approximation of ideal_f:n -> [0,1].
        Returns None
        Checks that self.f really is an approximation to the ideal, up to tolerance epsilon.
        """
        for x in range(2**self.num_working_qubits):
            assert np.allclose(self.f(x)/2**self.num_output_qubits, ideal_f(x)/(2*np.pi), atol=epsilon)
        self.check_guarantee()

    def apply_oracle(self,input_int) -> int:
        input_vect = Statevector.from_int(input_int,dims=2**self.num_working_qubits)
        zero_aux = Statevector.from_int(0,dims= 2**self.num_aux_qubits)
        out_vect = Statevector.from_int(0, dims = 2**self.num_output_qubits)
        initial_state = zero_aux.tensor(out_vect).tensor(input_vect)
        final_state = initial_state.evolve(self.circuit())
        outcome = final_state.measure(range(self.num_working_qubits,self.num_working_qubits+self.num_output_qubits))[0]
        return int(outcome[::-1],2)

class BinaryOracleFromLinear(BinaryOracle):
    """
    Create a BinaryOracle from a linear function Z_2^n -> Z_2^1
    """
    def __init__(self,num_working_qubits,bit_mask:int):
        #Assume bit_mask an integer whose binary representation are the coefficients of the linear function.
        def f(x):
            return (x & bit_mask).bit_count() % 2
        def circuit():
            working_qubits = QuantumRegister(num_working_qubits)
            output_qubit = QuantumRegister(1)
            qc = QuantumCircuit(working_qubits,output_qubit)
            for i, digit in enumerate(f"{bit_mask:0{num_working_qubits}b}"):
                if digit =='1':
                    qc.cx(working_qubits[num_working_qubits-1-i], output_qubit)
            return qc
        self.circuit = circuit
        super().__init__(num_working_qubits,1,0,f)
    def circuit(self):
        pass

class BinaryOracleFromLinearMultiOutput(BinaryOracle):
    """
    Create a BinaryOracle from a linear function Z_2^n -> Z_2^k
    """
    def __init__(self,num_working_qubits,num_output_qubits,bit_mask_list:list[int]):
        """Assume bit_mask a list of integers.
        The binary representation of the ith intger are the coefficients of the ith output of the linear function."""
        assert num_output_qubits == len(bit_mask_list)
        def f_digit(x, bit_mask):
            return (x & bit_mask).bit_count() % 2
        def f_factory(k):
            """
            input: A number of digits k
            ouput: A function that, on input x, returns number whose binary representation is given by f_digit.
            """
            def f(x):
                bits = [f_digit(x, bit_mask_list[i]) for i in range(k)]
                return int(''.join(map(str, bits)), 2)
            return f

        def circuit(k):
            """
            Input: a number of output qubits.
            Returns: a circuit
            """
            working_qubits = QuantumRegister(num_working_qubits)
            output_qubit = QuantumRegister(k)
            qc = QuantumCircuit(working_qubits,output_qubit)
            for out_index in range(len(output_qubit)):
                for i, digit in enumerate(f"{bit_mask_list[out_index]:0{num_working_qubits}b}"):
                    if digit =='1':
                        qc.cx(working_qubits[num_working_qubits-1-i], output_qubit[out_index])
            return qc
        self.circuit = lambda: circuit(num_output_qubits)
        super().__init__(num_working_qubits,num_output_qubits,0,f_factory(num_output_qubits))
    def circuit(self):
        pass

    
class BinaryOracleFromCircuit(BinaryOracle):
    
    def __init__(self, num_working_qubits,num_output_qubits,num_aux_qubits, circuit):
        def f(x):
            initial_vect =( Statevector.from_int(0,dims = 2**num_aux_qubits).tensor(Statevector.from_int(0,dims = 2**num_output_qubits))
                           .tensor(Statevector.from_int(x,dims=2**num_working_qubits)) )
            final_vect = initial_vect.evolve(circuit)
            outcome = final_vect.measure(range(num_working_qubits, num_working_qubits + num_output_qubits))[0]
            #outcome = final_vect.measure(range(num_working_qubits+num_aux_qubits,num_working_qubits+num_output_qubits + num_aux_qubits)[0] )
            return int(outcome[::-1],2)

        self.circuit = lambda: circuit
        super().__init__(num_working_qubits, num_output_qubits,num_aux_qubits,f)
    def circuit(self):
        pass
        
class BinaryOracleComparator(BinaryOracle):
    def __init__(self, num_working_qubits, threshold,maximize=True):
        def f(x):
            if maximize:
                if x >= threshold:
                    return 1
                else: 
                    return 0
            else:
                if x < threshold:
                    return 1
                else: 
                    return 0
        self.threshold = threshold

        circuit = IntegerComparator(num_state_qubits=num_working_qubits, value= self.threshold,geq = maximize)
        self.maximize = maximize
        num_aux_qubits = circuit.num_qubits - num_working_qubits -1
        super().__init__(num_working_qubits, 1,num_aux_qubits,f)
    def circuit(self):
        return IntegerComparator(num_state_qubits=self.num_working_qubits, value= self.threshold, geq = self.maximize)
    
class BinaryOracleMark(BinaryOracle):
    def __init__(self,num_working_qubits,mark):
        def f(x):
            if x == mark:
                return 1
            else:
                return 0
        def circuit():
            qc = QuantumCircuit(num_working_qubits+1)
            for qubit, bit in enumerate(f"{mark:0{num_working_qubits}b}"):
                if bit == '0':
                    qc.x(num_working_qubits -1-qubit)
            qc.mcx(qc.qubits[:-1],qc.qubits[-1])
            for qubit, bit in enumerate(f"{mark:0{num_working_qubits}b}"):
                if bit == '0':
                    qc.x(num_working_qubits -1-qubit)
            return qc
        self.circuit = circuit
        super().__init__(num_working_qubits, 1,0,f)
    def circuit(self):
        pass

class BinaryOracleHamming(BinaryOracle):
    def __init__(self, num_working_qubits, point:int):
        self.point = point
        num_output_qubits = int(1+math.floor(np.log2(num_working_qubits))) #see quskit WeightedAdder documentation

        def f(x:int) -> int:
            """
            Returns the hamming distance between x and point
            """
            return (x ^ point).bit_count()

        def circuit():
            #working_register = QuantumRegister(num_working_qubits)
            #output_register = QuantumRegister(num_output_qubits)

            adder_circuit = WeightedAdder(num_working_qubits)

            qc = QuantumCircuit(adder_circuit.num_qubits)
            for index, bit in enumerate(f"{point:0{num_working_qubits}b}"):
                if bit == '1':
                    qc.x(num_working_qubits-1-index)

            qc.append(adder_circuit,qargs = qc.qubits)

            for swap_gate_count in range(num_output_qubits//2):
                qc.swap(num_working_qubits+swap_gate_count,num_working_qubits + num_output_qubits-1-swap_gate_count)

            for index, bit in enumerate(f"{point:0{num_working_qubits}b}"):
                if bit == '1':
                    qc.x(num_working_qubits-1-index)

            return qc
        self.circuit = circuit
        super().__init__(num_working_qubits, num_output_qubits, self.circuit().num_qubits - num_working_qubits - num_output_qubits,f)
    def circuit(self):
        pass

