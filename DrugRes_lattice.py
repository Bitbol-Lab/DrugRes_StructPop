# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:01:41 2023

@author: cfruet

Use numba to accelerate the code.
The program aims at buiding the spatial structure of the lattice,
adding the antimicrobial at an user-defined time.

"""

#%%
from itertools import permutations
import numpy as np
from numpy import reshape
import math
import random
import sys
from numba import njit

#%%

@njit
def is_close_to_final_time(t, t_final, precision=1e-2):
    """
    Check if t is close to t_final within the specified precision.

    Parameters:
    - t: The value to be checked.
    - t_final: The target value.
    - precision: The allowed absolute difference.

    Returns:
    - True if |t - t_final| < precision, False otherwise.
    """
    return abs(t - t_final) < precision

#%%

def get_possible_migrations(node):
    '''
    Get the numbers indicating the places in the lattice where you can migrate, starting from node i. 
    Boundary conditions: periodic.

    Parameters
    ----------
    node : int
        number of the node under consideration.

    Returns
    -------
    possible_migrations : list
        number of the nodes to which you can migrate.

    '''
    i, j = divmod(node, 4)  # Convert node number to row and column indices
    possible_migrations = []

    # Add possible migrations for interior nodes
    if i > 0:
        possible_migrations.append(node - 4)  # Migrate to node above
    if i < 3:
        possible_migrations.append(node + 4)  # Migrate to node below
    if j > 0:
        possible_migrations.append(node - 1)  # Migrate to node on the left
    if j < 3:
        possible_migrations.append(node + 1)  # Migrate to node on the right

    # Handle nodes on the edges
    if j == 0:
        possible_migrations.append(node + 3)  # Migrate to node on the right edge of the same row
    elif j == 3:
        possible_migrations.append(node - 3)  # Migrate to node on the left edge of the same row

    if i == 0:
        possible_migrations.append(node + 12)  # Migrate to node on the bottom row
    elif i == 3:
        possible_migrations.append(node - 12)  # Migrate to node on the top row

    return possible_migrations

# Example usage

migrations = {}
for node in range(16):
    migrations[node] = get_possible_migrations(node)

#%%
#@njit
def build_Vmatrix_lattice(D):
    '''
    Build the reaction matrix for a lattice of demes
    
    Parameters
    ----------
    D  : int
        number of demes in the system.

    Returns
    -------
    V : matrix
        matrix of systems' reactions.

    '''
    # reactions in deme: D demes, 5 reactions,2 species
    rd = 5
    reacts_indeme = D * rd 
    md = D-1
    migrations = 4 * D
    tot_reacts = reacts_indeme + migrations * 2
    
    V = np.zeros((2*D, tot_reacts))
    
    # This is inside the single demes (5 reactions for each deme)
    l=0
    for i in range(D):
        V[2*i][0 + l*rd] = 1
        V[2*i][1 + l*rd] = -1
        V[2*i+1][2 + l*rd] = 1
        V[2*i+1][3 + l*rd] = 1
        V[2*i+1][4 + l*rd] = -1
        l+=1
    
    # Set the -1 for all reactions
    n = 0
    for m in range(D):
        for k in range(4):
            V[2*m][reacts_indeme + k + n*4] = -1
            V[2*m+1][reacts_indeme + migrations + k + n*4] = -1
        n+=1
    
    for k in range(16):
        tomig = get_possible_migrations(k)
        for i in range(4):
            V[2*tomig[i]][reacts_indeme + k*4 + i] = 1

    for k in range(16):
        tomig = get_possible_migrations(k)
        for i in range(4):
            V[2*tomig[i]+1][reacts_indeme + migrations + k*4 + i] = 1    

    return V


@njit
def simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, js0, jr0, t_final, n):
    extinctions_R = []
    fixations_R = []
    state_beforeAM = []
    final_state = []
    ks =[]
    count_fix_beforeAM = 0
    V = np.asarray(V, dtype=np.int64)
    X = np.zeros((2*n,1), dtype=np.float64)
    a = np.zeros((208,1), dtype=np.float64)


    for k in range(realisations):
        print('k=' + str(k))
        antimicrobial_added = False

        if k % 50 == 0:
            print('k=' + str(k))

        for i in range(16):
            X[2*i] = js0
            X[2*i + 1] = jr0

        t = 0
        fse = fs
        gse = gs

        ## ++++++++++++++++++ Second loop (over time) ++++++++++++++++++ ##
        while(t <= t_final):
            for y in range(16):
                a[0+ y*5] = fse * (1 - mi1) * (1 - (X[0 + y*2] + X[1 + y*2]  )/ K ) * X[0 + y*2] 
                a[1+ y*5] = gse * X[0+ y*2]
                a[2+ y*5] = fse * mi1 * (1 - (X[0+ y*2] + X[1+ y*2]  )/ K ) * X[0+ y*2] 
                a[3+ y*5] = fr * (1 - (X[0+ y*2] + X[1+ y*2]  )/ K ) * X[1+ y*2] 
                a[4+ y*5] = gr * X[1+ y*2]
            for g in range(16):
                for h in range(4):
                    a[80 + 4*g + h] = gamma * X[2*g]
            for g in range(16):
                for h in range(4):
                    a[144 + 4*g + h] = gamma * X[2*g+1]
                    
            for j in range(16):
                if (X[0 + j*2] + X[1 + j*2] >= K): 
                    #print('N exceeds K in deme ' + str(j))
                    a[0+ j*5] = 0
                    a[3+ j*5] = 0
                    
            asum = np.sum(a)
        
            cumulative = np.cumsum(a) / asum
            
            # ------ Random numbers extraction ------ #
            xi = np.random.rand(2)
            
            # --------- update time and check if time is higher than the antibiotic switching time --------- #
            tau = (np.log(1 / xi[1]) / asum)
            
            # --------- A) update time and check which reaction happens --------- #
            if (t + tau ) < Tadd or ((t + tau ) >= Tadd and antimicrobial_added):
               
                t = t + tau # update time normally :)
                
                j = []
                for i in range(len(cumulative)):
                    if xi[0] < cumulative[i] and cumulative[i] != 0:
                        j.append(i)
            
                j_int = int(min(j)) # as the index for V must be integer
                v_sel = V[:, j_int]
                v_sel_contig = np.ascontiguousarray(v_sel)
                
                new_shape = (32, 1)
            
                v_sel_contig = v_sel_contig.reshape(new_shape)
               
                X = X +v_sel_contig
                
                if X[0] == 0 and X[1] == 0 and X[2] == 0 and X[3] == 0 and X[4] == 0 and X[5] == 0 and X[6] == 0 and X[7] == 0 and X[8] == 0 and X[9] == 0 and X[10] == 0 and X[11] == 0 and X[12] == 0 and X[13] == 0 and X[14] == 0 and X[15] == 0 and X[16] == 0 and X[17] == 0 and X[18] == 0 and X[19] == 0 and X[20] ==0 and X[21] == 0 and X[22] == 0 and X[23] ==0 and X[24] == 0 and X[25] == 0 and X[26] ==0 and X[27] ==0 and X[28] == 0 and X[29] ==0 and X[30] == 0 and X[31] == 0:
                    print('population extict')
                    #print(X)
                    ks.append(k)
                    final_state.append(X.copy())
                    extinctions_R.append(t)
                    if len(final_state) != len(state_beforeAM):
                        state_beforeAM.append(X.copy())
                    break
                
                if X[0] == 0 and X[1] >= 0.9* EP and X[2] == 0 and X[3] >= 0.9* EP and X[4] == 0 and X[5] >= 0.9* EP and X[6] == 0 and X[7] >= 0.9* EP and X[8] == 0 and X[9] >= 0.9* EP and X[10] == 0 and X[11] >= 0.9* EP and X[12] == 0 and X[13] >= 0.9* EP and X[14] == 0 and X[15] >= 0.9* EP and X[16] == 0 and X[17] >= 0.9* EP and X[18] == 0 and X[19] >= 0.9* EP and X[20] == 0 and X[21] >= 0.9* EP and X[22] == 0 and X[23] >= 0.9 * EP and X[24] == 0 and X[25] >= 0.9 *EP and X[26] == 0 and X[27] >= 0.9 *EP and X[28] == 0 and X[29] >= 0.9 * EP and X[30] == 0 and X[31] >= 0.9*EP:
                    print('population survived')
                    ks.append(k)
                    final_state.append(X.copy())
                    fixations_R.append(t)
                    if len(final_state) != len(state_beforeAM):
                        print('pop fix before AM')
                        state_beforeAM.append(X.copy())
                        count_fix_beforeAM += 1
                    break
            
                if is_close_to_final_time(t, t_final, 1e-3):
                    print("t is close to t_final within the specified precision.")
                    print('tfinal')
                    ks.append(k)
                    final_state.append(X.copy())
                    break
                    
            # --------- B) set switch and change rates --------- # 
            elif (t + tau ) >= Tadd and not antimicrobial_added:
                print("tadd reached")
                t = Tadd
                state_beforeAM.append(X.copy())
                
                fse = fsp
                gse = gsp
                print('Add antimicrobial')
                antimicrobial_added = True

    return ks, count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state



#%% Parameters
V = build_Vmatrix_lattice(16)

fs = 1 
gs = 0.1
fsp = 0
gsp = 0.1 

fr = 1 
gr = 0.1 

gamma= ( 15/4 ) *  1e-7
mi1 = 1e-5

K = 100

js0 = 10 
jr0 = 0 

n = 16 # number of demes, spatial structure

Tadd = 500000

realisations = 100 # Number of simulations (=runs)

t = 0
t_final = 1000000000
EP = (1 - gs / fs) * K

#%%


ks, count_fix_beforeAM,  extinctions_R, fixations_R, state_beforeAM, final_state_end = simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, js0, jr0, t_final, n)

#%%

print(sys.argv)
i = int(sys.argv[1])
print(i)

output_surv = f'survival_LATTICE_T{Tadd}_{i}.npy'
output_ext = f'extinction_LATTICE_T{Tadd}_{i}.npy'
output_state = f'state_LATTICE_beforeAM_T{Tadd}_{i}.npy'
final_state = f'state_LATTICE_final_T{Tadd}_{i}.npy'
count_fix_bAM = f'count_fix_bAM_LATTICE_T{Tadd}_{i}.npy'

np.save(output_state, state_beforeAM)
np.save(output_surv, fixations_R)
np.save(output_ext, extinctions_R)
np.save(final_state, final_state_end)
np.save(count_fix_bAM, count_fix_beforeAM)


