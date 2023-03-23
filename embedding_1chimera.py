import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector as inspector
from dimod import BinaryQuadraticModel as BQM

def manual_embedding(qubit_biases, coupler_strengths, shots = 2):

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    Q = {**qubit_biases, **coupler_strengths}
    sampleset = sampler_manual.sample_qubo(Q, num_reads=shots)
    print(sampleset)
    inspector.show(sampleset)

def auto_embedding(matrix, shots = 1): 

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    embedding = EmbeddingComposite(sampler_manual)
    running = embedding.sample_qubo(matrix, num_reads=shots)
    print(running)
    inspector.show(running)


# Pasamos el problema de particiones con 3 y 4 variables. Asi entra completo en una chimera

set = np.array([1, 2, 3, 4])
size = np.size(set)
suma = sum(set)

qubo = np.outer(2*set, 2*set) - 2*suma*np.diag(2*set)

# El peso de las cadenas lo calculamos poniendo una penalizacion a (x - y)^2. 
# Elegimos una penalizacion P igual al maximo (en valor absoluto) de los pesos de la matriz qubo
# Se podria estudiar luego los efectos de distintas penalizaciones. O tomar penalizaciones especificas
# Para pares especificos de variables i, j.

# P = np.max(np.absolute(qubo))
P = np.mean(np.absolute(qubo))

dimentions = 0
# Creamos un diccionario con el peso de cada variable y las cadenas:
qubit = dict()
chains = dict()
for i in range(0, size):
    # i ---> variable
    qubit[(i, i)] = 0.5 * qubo[i, i] + P
    qubit[(i + 4, i + 4)] = 0.5 * qubo[i, i] + P
    chains[(i, i + 4)] = -2*P
    
for i in range(0, size):
    for j in range(0, size):
        if(i != j):
            chains[(i, j + 4)] = qubo[i, j]



qubit_biases = qubit
coupler_strengths = chains

print(qubit_biases)
print(coupler_strengths)


manual_embedding(qubit_biases, coupler_strengths, 30)
# auto_embedding(qubo, 30)
