import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector as inspector
from dimod import BinaryQuadraticModel as BQM

# corre el qubo con un embedding manual
def manual_embedding(qubit_biases, coupler_strengths, shots = 2):
    # qubit_biases ---> python dict con pares {(qubit 1, qubit 1) = bias qubit 1}
    # coupler_strenghts ---> python dict con pares {(qubit 1, qubit 2) = coupler strenght entre 1 y 2}
    # LOS BIAS Y LOS COUPLER STRENGHTS ESTAN EN MODELO QUBO
    # shots ---> cantidad de veces a correr el problema

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    Q = {**qubit_biases, **coupler_strengths}
    sampleset = sampler_manual.sample_qubo(Q, num_reads=shots)
    return sampleset


# corre el qubo con un embedding automatico
def auto_embedding(matrix, shots = 1): 
    # matrix ---> matriz QUBO del problema
    # shots ---> cantidad de veces a correr el problema

    sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})
    embedding = EmbeddingComposite(sampler_manual)
    running = embedding.sample_qubo(matrix, num_reads=shots)
    return running