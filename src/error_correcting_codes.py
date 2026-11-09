from search import *
from oracle_conversions import *
class LinearCode(object):
    def __init__(self, H=None, G=None):
        """
        Following the notation of Macwilliams and Sloane.
        Assume H is a (n-k) x n matrix of the form H =[A| I_{n-k}]. It is the parity check matrix. H^T c=0 for codewords c.
        Assume G is a k x n matrix of the form G= [I_{k}|-A^T]. We generate codewords by c=uG.
        """
        if H is not None:
            self.H = H
            self.n = len(H[0])
            self.k = self.n-len(H)
            self.A = [ row[:self.k]for row in H]
            Ik = [[ 1 if row_index == col_index else 0 for col_index in range(self.k)] for row_index in range(self.k)]
            self.neg_At = [ [-self.A[col_index][row_index] for col_index in range(len(self.A))] for row_index in range(len(self.A[0]))]
            self.G = [ Ik[row] + self.neg_At[row] for row in range(self.k)]
        elif G is not None:
            self.G=G
            self.n = len(G[0])
            self.k = len(G)
            self.neg_At = [ row[self.k:] for row in G]
            self.A = [ [-self.neg_At[col_index][row_index] for col_index in range(len(self.neg_At))] for row_index in range(len(self.neg_At[0]))]
            Inmk = [[ 1 if row_index == col_index else 0 for col_index in range(self.n-self.k)] for row_index in range(self.n-self.k)]
            self.H = [ self.A[row_index] + Inmk[row_index] for row_index in range(self.n-self.k)]
    def generation_binary_oracle(self):
        """
        Returns a binary oracle where the input is assumed to be |x>|0>, and then the output is the basis state of a codeword |c>
        """
        bit_mask_list = [int("".join(str(r%2) for r in row[::-1]),2) for row in self.A]
        return BinaryOracleFromLinearMultiOutput(self.k,self.n-self.k,bit_mask_list)
    
    def encode_classical(self, logical_bit_string:str)->str:
        logical_bit_vector = [[int(bit) for bit in logical_bit_string[::-1] ]]
        physical_bit_vector = np.array(logical_bit_vector)@np.array(self.G) % 2
        physical_bit_string = "".join([str(row % 2) for row in physical_bit_vector[0]])
        return physical_bit_string

    def decode_classical(self, observed_bit_string:str)->str:
        """
        input: observed_bit_string is the observed bit string (length self.n).
        returns: the logical word whose encoding is as close to observed_bit_string as possible.
        """
        assert len(observed_bit_string)==self.n
        best_distance = math.inf
        best_candidate = None
        for candidate in range(2**self.k):
            encoded_candidate = self.encode_classical(f"{candidate:0{self.k}b}")
            hamming_distance = (int(encoded_candidate,2 )^ int(observed_bit_string,2)).bit_count()
            if hamming_distance < best_distance:
                best_candidate = candidate
                best_distance = hamming_distance
        return f"{best_candidate:0{self.k}b}"
    
    def decode_quantum(self, observed_bit_string:str)->str:
        """
        input: observed_bit_string is the observed bit string (length self.n).
        returns: the logical codeword 
        Note: Applies Durr-hoyer search
        """
        working_register = QuantumRegister(self.k)
        initial_unitary = QuantumCircuit(working_register)
        initial_unitary.h(working_register)
        point = int(observed_bit_string,2)
        #hamming_dist = compose_binary_oracles(encoding_oracle, BinaryOracleHamming(encoding_oracle.num_output_qubits,point))
        hamming_dist = BinaryOracleECCHamming(self,point)
        return dur_hoyer_search(initial_unitary, hamming_dist, maximize=False, prob_failure=0.1)

class ExampleLinearCode(LinearCode):

    def __init__(self):
        parity_check_matrix = [[0,1,1,1,1,0,0],
                                [1,0,1,1,0,1,0],
                                [1,1,0,1,0,0,1]]
        super().__init__(H= parity_check_matrix)

class ExampleLinearCode2(LinearCode):
    """
    This [9,3,4] linear code comes from https://docs.magma-maths.org/CodingTheory/LinearCodes/best-codes.html
    """
    def __init__(self):
        G = [[1, 0, 0, 1, 0, 1, 1, 0, 0],
             [0, 1, 0, 1, 0, 1, 0, 1, 0],
             [0, 0, 1, 1, 0, 0, 1, 1, 0],]
        super().__init__(G=G)

class BinaryOracleECCHamming(BinaryOracle):
    def __init__(self,E:LinearCode, point:int):
        self.point = point
        self.E=E
        num_input_hamming_qubits =  E.n
        num_output_hamming_qubits = int(1+math.floor(np.log2(num_input_hamming_qubits))) #see quskit WeightedAdder documentation
        self.num_working_qubits=E.k
        self.num_output_qubits=num_output_hamming_qubits

        def f(x:int) -> int:
            """
            Returns the hamming distance between x and point
            """
            return (int(E.encode_classical(f"{x:0{E.k}b}"),2) ^ point).bit_count()
        def circuit():

            adder_circuit = WeightedAdder(num_input_hamming_qubits)

            qc1 = E.generation_binary_oracle().circuit()
            qc = QuantumCircuit(adder_circuit.num_qubits)
            for qbt in range(E.n, E.n+ num_output_hamming_qubits):
                qc.swap(qbt,qbt-(E.n-E.k))

            qc.append(qc1, qargs = range(E.n))


            for index, bit in enumerate(f"{point:0{E.n}b}"):
                if bit == '1':
                    qc.x(index)

            qc.append(adder_circuit, qargs = qc.qubits)

            for index, bit in enumerate(f"{point:0{E.n}b}"):
                if bit == '1':
                    qc.x(index)
            for qbt in range(E.n+ num_output_hamming_qubits-1,E.n-1,-1):
                qc.swap(qbt,qbt-(E.n-E.k))

            for qbt in range(num_output_hamming_qubits//2):
                qc.swap(E.k+qbt, E.k+ num_output_hamming_qubits-qbt-1)

            return qc
        self.circuit = circuit
        self.num_aux_qubits = self.circuit().num_qubits - self.num_working_qubits -self.num_output_qubits

        super().__init__(self.num_working_qubits, self.num_output_qubits, self.num_aux_qubits,f)
    def circuit(self):
        pass
