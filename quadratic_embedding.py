import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector as inspector
from mapping_qubits import *

def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result


def auto_embedding(matrix): 

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    embedding = EmbeddingComposite(sampler_manual)
    running = embedding.sample_qubo(matrix, num_reads=1000)
    print(running)
    inspector.show(running)

def auto_embedding_chain(matrix, chain = 1): 

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    embedding = EmbeddingComposite(sampler_manual)
    running = embedding.sample_qubo(matrix, num_reads=1000, chain_strength = chain)
    print(running)
    inspector.show(running)

def manual_embedding(qubit_biases, coupler_strengths, shots = 2):

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    Q = {**qubit_biases, **coupler_strengths}
    sampleset = sampler_manual.sample_qubo(Q, num_reads=shots)
    print(sampleset)
    inspector.show(sampleset)


set = np.array([1, 2, 3, 4, 9, 11, 15, 20])
size = np.size(set)
suma = sum(set)
dimentions = int(np.floor(size/4)) # Cantidad de chimeras diagonales a usar (arrancando en cero)


qubo = np.outer(2*set, 2*set) - 2*suma*np.diag(2*set)
print(qubo[1, :])


variables = size - 1
shift = 0 # Arrancamos en el qubit 16*8
Q = dict()
C = dict()


for i in range(0, variables + 1):
    map = qubits_dict(i, variables, qubo, shift)
    qubits = map[0]
    chains = map[1]

    Q = Merge(Q, qubits)
    C = Merge(C, chains)

manual_embedding(Q, C, 1000)
# auto_embedding(qubo)


'''
for i in range(0, variables + 1):
    map = triangular_embedding(i, variables, qubo, shift)
    qubits = map[0]
    chains = map[1]

    Q = Merge(Q, qubits)
    C = Merge(C, chains)

manual_embedding(Q, C, 1000)
# auto_embedding_chain(qubo, chain = np.max(np.absolute(qubo)))
'''