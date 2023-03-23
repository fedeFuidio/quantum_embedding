import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector as inspector
from dimod import BinaryQuadraticModel as BQM

# p = 4, n = 16

def chimera_column(k):
    return int(np.floor(k/8) % 16)

def chimera_row(k):
    return int(np.floor(k/(8*16)))

def qubit_column(k):
    return int(4*chimera_column(k) + (k % 4))

def qubit_row(k):
    return int(4*chimera_row(k) + (k % 4))


# La variable n del problema se mapea a los qbits diagonales
def qubits_variable(n):
    chimera = int(np.floor(n/4)) # Nos dice el numero de la chimera diagnoal (chimera, chimera)
    qubit_chimera = int(n) % 4 # Vamos a usar qubit_chimera y qubit_chimera + 4 como diagonal
    qubit_number = chimera * (8*16) + 8*(chimera % 8)

    return qubit_number + qubit_chimera



def get_variable(qubit, shift = 0):
    # qubit ---> numero del qubit en D'wave
    # shift ---> shift inicial del embedding

    qubit_mod8 = qubit % 8
    qubit_mod4 = qubit % 4
    if(qubit_mod8 <= 3):
        # El qubit es vertical
        chimera_column = int(np.floor((qubit % (8*16))/8))
        d = chimera_column

    else:
        # El qubit es horizontal
        chimera_row = int(np.floor(qubit/(8*16)))
        d = chimera_row


    return d*4 + (qubit_mod4)

    

def add_horizontal_edges(n, qubit, edges, n_mod4, chains, qubo):
    # edges: del 1 al 4

    for i in range(0, edges):
        variable = get_variable(qubit - n_mod4 - (4 - i))
        chains[(qubit - n_mod4 - (4 - i), qubit)] = qubo[n, variable]
    
    return chains

def add_vertical_edges(n, qubit, edges, n_mod4, chains, qubo):
    # edges: del 1 al 4
    # n ---> variable
    # qubit ---> qubit que esta en la cadena de la variable n

    for i in range(0, edges):
        variable = get_variable(qubit - n_mod4 + (4 + i))
        chains[(qubit, qubit - n_mod4 + (4 + i))] =  qubo[n, variable]
    
    return chains



# QUADRATIC EMBEDDING
def qubits_dict(n, variables, qubo, shift = 0): 
    # n ---> numero de variable en el problema contando desde 0. (x_0, x_1, ..., [x_n], x_{n+1}, ...)
    # variables ---> Maximo valor de los indices de variabla. (la lista de arriba va hasta x_{variables})
    # shift ---> chimera inicial. Numerando chimeras como d'wave. El qubit inicial es (shift * 8)

    # Retorna la chain a la que se mapea el vertice n del grafo qubo en los chimeras.

    chimera = int(np.floor(n/4)) # Chimera diagonal donde se unen la cadena horizonal y vertical para la variable n
    max_chimera = int(np.floor(variables/4)) # maxima chimera diagonal (Nos dice donde cortar la cadena)
    
    chain_len = 2*(max_chimera + 1) 
    chain_weight = 1/(chain_len)
    
    P_mean = np.max(np.absolute(qubo[n, :])) # Peso que agregamos entre elementos de la cadena

    # n debe ser a lo sumo igual a la cantidad de variables totales
    if(n > variables):
        return

    n_mod4 = n % 4 # Vamos a usar varias veces la variable, esta bueno tenerla.
    
    # Qubits de inicio
    Vhori = (shift * 8) + chimera*(8*16) + n_mod4 + 4
    Vvert = (shift * 8) + chimera*(8) + n_mod4

    qubits_hori = []
    qubits_vert = []
    qubits = dict()
    chains = dict()

    P_set = 0 # Seteamos una variable en cero.
    if(n == 0):
        #P_set = 2*np.max(np.absolute(qubo))
        P_set = np.max(np.absolute(qubo))

    # contruimos las cadenas horizontales y verticales. Las unimos luego del for
    for i in range(0, max_chimera + 1):

        qubits_hori.append(Vhori + (8*i)) 
        qubits_vert.append(Vvert + (8*16)*i)
        
        # Si estamos en un endpoint un qubit esta unido a uno de la cadena (no a dos como los demas). por eso
        # el peso es P_mean.
        if (((i == max_chimera and chimera != max_chimera) or (i == 0 and chimera != 0)) or max_chimera == 0):
        
            qubits[(qubits_hori[i], qubits_hori[i])] = chain_weight*qubo[n, n] + P_mean + P_set
            qubits[(qubits_vert[i], qubits_vert[i])] = chain_weight*qubo[n, n] + P_mean + P_set

        else:
            qubits[(qubits_hori[i], qubits_hori[i])] = chain_weight*qubo[n, n] + 2*P_mean + P_set
            qubits[(qubits_vert[i], qubits_vert[i])] = chain_weight*qubo[n, n] + 2*P_mean + P_set
            
        
        # Si no estamos en la ultima chimera agregamos todas las combinaciones posibles en la chimera
        if(i < max_chimera):
            chains = add_horizontal_edges(n, qubits_hori[i], 4, n_mod4, chains, qubo)
            chains = add_vertical_edges(n, qubits_vert[i], 4, n_mod4, chains, qubo)

        # En la ultima chimera agregamos solo las conexiones hasta donde llega la ultima variable
        else:
            chains = add_horizontal_edges(n, qubits_hori[i], (variables % 4), n_mod4, chains, qubo)
            chains = add_vertical_edges(n, qubits_vert[i], (variables % 4), n_mod4, chains, qubo)

        # conectamos las cadena
        if (i > 0):
            chains[(qubits_hori[i], qubits_hori[i-1])] = -2*P_mean
            chains[(qubits_vert[i], qubits_vert[i-1])] = -2*P_mean



    # conectamos la cadena vertical con la horizontal (los que estan en la misma chimera)
    chains[(qubits_vert[chimera], qubits_hori[chimera])] = -2*P_mean

    return [qubits, chains]


# TRIANGULAR EMBEDDING (triangular inferior <-----> triangular superior):
def triangular_embedding(n, variables, qubo, shift = 0):
    # n ---> numero de variable en el problema (x_0, x_1, ..., [x_n], x_{n+1}, ...)
    # variables ---> Maximo valor de los indices de variables. la lista de arriba va hasta x_{variables}
    # shift ---> chimera inicial. Numerando chimeras como d'wave. El qubit inicial es (shift * 8)

    chimera = int(np.floor(n/4)) # Chimera diagonal donde se unen la cadena horizonal y vertical
    max_chimera = int(np.floor(variables/4)) # indice de maxima chimera diagonal
    chain_len = (max_chimera + 2)

    chain_weight = 1/(chain_len)
    P_mean = 1.5*np.max(np.absolute(qubo))

    P_set = 0 # Seteamos una variable en cero.
    if(n == 0):
        P_set = 2*np.max(np.absolute(qubo))

    if(n > variables):
        return

    n_mod4 = n % 4 # Vamos a usar varias veces la variable, esta bueno tenerla.
    # Qubits de inicio
    Vhori = (shift * 8) + chimera*(8*16) + n_mod4 + 4
    Vvert = (shift * 8) + chimera*(8) + n_mod4

    qubits_hori = []
    qubits_vert = []
    qubits = dict()
    chains = dict()

    # Agregamos la cadena horizontal primero (ahora va de 0 hasta la chimera diagonal): 
    for i in range(0, chimera + 1):
        qubits_hori.append(Vhori + (8*i)) 

        # Si estamos en un endpoint
        if(i == 0 and 0 != chimera):
            qubits[(qubits_hori[i], qubits_hori[i])] = chain_weight*qubo[n, n] + P_mean

        else:
            qubits[(qubits_hori[i], qubits_hori[i])] = chain_weight*qubo[n, n] + 2*P_mean
        

        if(i < max_chimera):

            # Las conexiones con variables en la misma chimera son dobles. Las demas son simples
            if(i == chimera):
                chains = add_horizontal_edges(n, qubits_hori[i], 4, n_mod4, chains, qubo)

            else:
                chains = add_horizontal_edges(n, qubits_hori[i], 4, n_mod4, chains, 2*qubo)


        else:

            # Las conexiones con variables en la misma chimera son dobles. Las demas son simples
            if (i == chimera):
                chains = add_horizontal_edges(n, qubits_hori[i], (variables % 4), n_mod4, chains, qubo)

            else:
                chains = add_horizontal_edges(n, qubits_hori[i], (variables % 4), n_mod4, chains, 2*qubo)
    
       
        if (i > 0):
            chains[(qubits_hori[i], qubits_hori[i-1])] = -2*P_mean
            



    # Agregamos la cadena vertical (de la diagonal hasta la chimera final)
    for i in range(chimera, max_chimera + 1):
        qubits_vert.append(Vvert + (8*16)*i)

        # Si estamos en un endpoint
        if(i == max_chimera and chimera != max_chimera):
            qubits[(qubits_vert[i - chimera], qubits_vert[i - chimera])] = chain_weight*qubo[n, n] + P_mean


        else:
            qubits[(qubits_vert[i - chimera], qubits_vert[i - chimera])] = chain_weight*qubo[n, n] + 2*P_mean
        


        if(i < max_chimera):
            if (i == chimera):
                chains = add_vertical_edges(n, qubits_vert[i - chimera], 4, n_mod4, chains, qubo)

            else:
                chains = add_vertical_edges(n, qubits_vert[i - chimera], 4, n_mod4, chains, qubo)

        else:
            if (i == chimera):
                chains = add_vertical_edges(n, qubits_vert[i - chimera], (variables % 4), n_mod4, chains, qubo)

            else:
                chains = add_vertical_edges(n, qubits_vert[i - chimera], (variables % 4), n_mod4, chains, qubo)

        if (i - chimera > 0):
            chains[(qubits_vert[i - chimera], qubits_vert[i - chimera-1])] = -2*P_mean



    # conectamos tambien los elementos Floor(n/4) (los que estan en la misma chimera)

    chains[(qubits_vert[0], qubits_hori[chimera])] = -2*P_mean

    return [qubits, chains]