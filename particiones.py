import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector as inspector
from dimod import BinaryQuadraticModel as BQM
from mapping_qubits import *

def manual_embedding(qubit_biases, coupler_strengths, shots = 2):

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    Q = {**qubit_biases, **coupler_strengths}
    sampleset = sampler_manual.sample_qubo(Q, num_reads=shots)
    print(sampleset)
    inspector.show(sampleset)

set = np.array([1, 2, 3, 4])
size = np.size(set)
suma = sum(set)
dimentions = int(np.floor(size/4)) # Cantidad de chimeras diagonales a usar (arrancando en cero)


qubo = np.outer(2*set, 2*set) - 2*suma*np.diag(2*set)

# El peso de las cadenas lo calculamos poniendo una penalizacion a (x - y)^2. 
# Elegimos una penalizacion P igual al maximo (en valor absoluto) de los pesos de la matriz qubo
# Se podria estudiar luego los efectos de distintas penalizaciones. O tomar penalizaciones especificas
# Para pares especificos de variables i, j.

P_max = np.max(np.absolute(qubo))
P_mean = np.mean(np.absolute(qubo))

dimentions = 0
# Creamos un diccionario con el peso de cada variable y las cadenas:
qubit = dict()
chains = dict()
for i in range(0, size):
    # i ---> variable
    # q ---> lista con todos los qubits fisicos que representa
    q = qubits_variable(i)
    qubit[(q, q)] = 0.5 * qubo[i, i] + P_mean
    qubit[(q + 4, q + 4)] =  0.5 * qubo[i, i] + P_mean
    chains[(q, q+4)] = - 2*P_mean
    if(dimentions == 1):
        qubit[(q + 4 + 8, q + 4 + 8)] =  0.5 * qubo[i, i] + P_mean
        qubit[(q + 8*16, q + 8*16)] = 0.5*qubo[i, i] + P_mean
        chains[(q + 4, q + 4 + 8)] = qubo[i, i]
        chains[(q, q + 8*16)] = qubo[i, i]
    

# qubit[(0, 0)] = qubit[(0, 0)] + P_max
# qubit[(4, 4)] = qubit[(4, 4)] + P_max # Obligamos al a_1 a estar en el segundo conjunto

for i in range(0, size):
    q_i = qubits_variable(i)
    for j in range(0, size):
        q_j = qubits_variable(j)
        if(q_i != q_j):
            chains[(q_i, q_j+4)] = qubo[i, j]



qubit_biases = qubit
coupler_strengths = chains


manual_embedding(qubit_biases, coupler_strengths, 10)
