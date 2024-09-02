# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:01:41 2023

@author: cfruet

Use numba to accelerate the code.
The program aims at buiding the spatial structure of the star,
adding the antimicrobial at an user-defined time.


"""

#%%

import numpy as np
from numpy import reshape
import math
import random
import sys
from numba import njit



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
@njit
def build_Vmatrix_star(D):
    '''
    Build the reaction matrix for a star. As for the numbering, deme 1 is the center.
    Then, all other demes are numbered according to clockwise direction, starting from the top
    (even if at this point all leaves are equivalent - > the numbering of the leaves does not matter)
    
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
    migrations_fromC = D-1
    migrations_toC = D-1
    tot_reacts = reacts_indeme + migrations_fromC * 2 + migrations_toC * 2
    
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
    
    #migrations from the center to the leaves, for both wild-types and mutants
    for n in range(D-1):
        V[0][reacts_indeme +n] = -1
        V[1][reacts_indeme +(D-1) + n] = -1

    for m in range(1, D):
        V[2*m][reacts_indeme + m-1] = 1
        V[2*m + 1][reacts_indeme + ( D-1) + m-1] = 1
        
    # Migrations from the leaves to the center
    for n in range(D-1):
        V[0][reacts_indeme + migrations_fromC * 2 +n] = 1
        V[1][reacts_indeme + migrations_fromC * 2 + (D-1) + n] = 1
        
    for m in range(1, D):
        V[2*m][reacts_indeme + migrations_fromC + migrations_toC + m-1] = -1
        V[2*m + 1][reacts_indeme + migrations_fromC + migrations_toC + ( D-1) + m-1] = -1
    
    return V


@njit
def simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, alpha, js0, jr0, t_final, n):
    extinctions_R = []
    fixations_R = []
    state_beforeAM = []
    final_state = []
    D = n
    survival_threshold = 0.9 * EP
    count_fix_beforeAM = 0
    pres_mut_bAM = 0
    pres_mut_afterAM = 0
    tot_reacts = V.shape[1]
    V = np.asarray(V, dtype=np.int64)
    X = np.zeros((2*n,1), dtype=np.float64)
    a = np.zeros((tot_reacts ,1), dtype=np.float64)
    rd = 5
    reacts_indeme = D * rd 
    migrations_fromC = D-1
    migrations_toC = D-1


    for k in range(realisations):
        print('k=' + str(k))
        antimicrobial_added = False

        if k % 50 == 0:
            print('k=' + str(k))

        for i in range(D):
            X[2*i] = js0
            X[2*i + 1] = jr0

        t = 0
        fse = fs
        gse = gs

        ## ++++++++++++++++++ Second loop (over time) ++++++++++++++++++ ##
        while(t <= t_final):
        
            for y in range(D):
                a[0+ y*5] = fse * (1 - mi1) * (1 - (X[0 + y*2] + X[1 + y*2]  )/ K ) * X[0 + y*2] 
                a[1+ y*5] = gse * X[0+ y*2]
                a[2+ y*5] = fse * mi1 * (1 - (X[0+ y*2] + X[1+ y*2]  )/ K ) * X[0+ y*2] 
                a[3+ y*5] = fr * (1 - (X[0+ y*2] + X[1+ y*2]  )/ K ) * X[1+ y*2] 
                a[4+ y*5] = gr * X[1+ y*2]
                
            # migrations from the center (center has index 0 for S, 1 for R)
            for g in range(D-1):
                a[reacts_indeme + g] = gamma * X[0]
                # here below add migrations_fromC = (D-1) as it is the number of reactions from the center to the leaves
                a[reacts_indeme + migrations_fromC + g] = gamma * X[1]
                
            # migrations to the center
            for g in range(D-1):
                a[reacts_indeme + migrations_fromC*2 + g] = alpha * gamma * X[2*(g+1)]
                a[reacts_indeme + migrations_fromC*2 + migrations_toC + g] = alpha * gamma * X[2*(g+1)+1]
         
            for j in range(D):
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
                all_RSzero = True
                for i in range(0, len(X)):
                    if not (X[i] == 0):
                        all_RSzero = False
                        break

                if all_RSzero:
                    print('population extinct')
                    final_state.append(X.copy())
                    extinctions_R.append(t)
                    if len(final_state) != len(state_beforeAM):
                        state_beforeAM.append(X.copy())
                    break

            
                all_Szero = True
                for i in range(0, len(X), 2):
                    if not (X[i] == 0 and X[i+1] >= survival_threshold):
                        all_Szero = False
                        break
                   
                if all_Szero:
                    print('population survived')
                    final_state.append(X.copy())
                    pres_mut_afterAM += 1
                    fixations_R.append(t)
                    if len(final_state) != len(state_beforeAM):
                        print('pop fix before AM')
                        state_beforeAM.append(X.copy())
                        count_fix_beforeAM += 1
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

    return  count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state, pres_mut_bAM, pres_mut_afterAM



#%% Parameters
V = build_Vmatrix_star(16)

fs = 1 
gs = 0.1
fsp = 0
gsp = 0.1 

fr = 1
gr = 0.1 

gamma=  15 * 1e-6
mi1 = 1e-5
alpha = 0.25 # 1 # 0.25

K = 100

js0 = 10 
jr0 = 0 

n = 16 # number of demes, spatial structure

Tadd = 1500000

realisations = 20 # Number of simulations (=runs)

t = 0
t_final = 1000000000
EP = (1 - gs / fs) * K




count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state_end, pres_mut_bAM, pres_mut_aAM = simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, alpha, js0, jr0, t_final, n)



print(sys.argv)
i = int(sys.argv[1])
print(i)

output_surv = f'survival_STAR4_sameflux_alpha{alpha}_fr{fr}_T{Tadd}_{i}.npy'
output_ext = f'extinction_STAR4_sameflux_alpha{alpha}_fr{fr}_T{Tadd}_{i}.npy'
output_state = f'state_STAR4_sameflux_alpha{alpha}_fr{fr}_beforeAM_T{Tadd}_{i}.npy'
final_state = f'state_STAR4_sameflux_alpha{alpha}_fr{fr}_final_T{Tadd}_{i}.npy'
count_fix_bAM = f'count_fix_bAM_STAR4_sameflux_alpha{alpha}_fr{fr}_T{Tadd}_{i}.npy'

np.save(output_state, state_beforeAM)
np.save(output_surv, fixations_R)
np.save(output_ext, extinctions_R)
np.save(final_state, final_state_end)
np.save(count_fix_bAM, count_fix_beforeAM)

