# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:01:41 2023

@author: cfruet

Use numba to accelerate the code.
The program aims at buiding the spatial structure of the clique,
adding the antimicrobial at an user-defined time.

"""

#%%

import numpy as np
import sys
from numba import njit

#%%

@njit
def is_close_to_final_time(t, t_final, precision=1e-2):
    """
    Check if t is close to t_final within the specified precision.

    Parameters:
    ----------
    t : float
        The value to be checked.
    t_final : float
        The target value for the final time.
    precision : float
        The allowed absolute difference.

    Returns:
    ----------
        True if |t - t_final| < precision, False otherwise.
    """
    return abs(t - t_final) < precision



@njit
def build_Vmatrix_clique(D):
    '''
    Build the reaction matrix for a clique of demes
    
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
    migrations = (D-1) * D
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
        for k in range(D-1):
            V[2*m][reacts_indeme + k + n*(D-1)] = -1
            V[2*m+1][reacts_indeme + migrations + k + n*(D-1)] = -1
        n+=1
    
    #S0, R0 - first deme
    for j in range(D-1):
        V[2*j+2][reacts_indeme +j] =1

    for j in range(D):
        if (2*j+1 == 1):
            print(2*j + 1)
        else:
            V[2*j+1][reacts_indeme -1 +migrations +j] = 1
    
    list_perm = np.arange(D-1)
    new_perm = np.zeros((D-1,))
    
    # All other demes
    for p in range(1, D):
        
        for h in range(D-1):
            new_perm[h] = list_perm[(h+(D-1 - p))%(D-1)]
    
        count = 0
        for j in range(D):
            if 2*j == 2 * p:
                print(2*j)
            else: 
                V[2*j][reacts_indeme+ p * md + int(new_perm[count])] =1
                count+=1
                
        count = 0
        for j in range(D):
            if 2*j +1 == 2 *p +1 :
                print(2*j + 1)
            else: 
                V[2*j+1][reacts_indeme + migrations + p * md + int(new_perm[count%(D-1)])] =1
                count+=1   
    return V



@njit
def simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, js0, jr0, t_final, n):
    '''
    Gillespie simulation of the clique with addition of AM after Tadd

    Parameters
    ----------
    V : array-like
        stoichiometric matrix for the system.
    realisations : int
        number of realizations.
    Tadd : int
        addition time of antimicrobial.
    fs : float
        sensitive fitness.
    gs : float
        sensitive death rate.
    fsp : float
        sensitive fitness after AM.
    gsp : float
        sensitive death rate after AM.
    EP : int
        equilibirum population size.
    K : int
        carrying capacity.
    mi1 : float
        mutation rate from sensitive to resistant.
    fr : float
        resistant fitness.
    gr : float
        resistant death rate.
    gamma : float
        migration rate.
    js0 : int
        initial sensitive population.
    jr0 : int
        initial resistant population.
    t_final : int
        final time to stop simulation.
    n : int
        number of demes in the system.

    Returns
    -------
    count_fix_beforeAM : int
        counts the times the mutants fixed before addition of the antimicrobial.
    extinctions_R : list
        stores the times at which the R-types and S-types got extinct (all system has zero individuals).
    fixations_R : list
        stores the times at which the R-types fixated in all population (ultimate fixation, in all demes).
    state_beforeAM : array-like
        stores the system composition before drug application.
    final_state : array-like
        stores the final state (extinction/fixation), just as a dobule check.
    pres_mut_bAM : int
        counts the number of times there are mutants before the AM application.
    pres_mut_afterAM : int
        counts the number of times there are mutants after the AM application.

    '''
    extinctions_R = []
    fixations_R = []
    state_beforeAM = []
    final_state = []
    
    survival_threshold = 0.9 * EP
    count_fix_beforeAM = 0
    pres_mut_bAM = 0
    pres_mut_afterAM = 0
    
    # reactions numbers (to populate the propensities)
    n_r = V.shape[1]
    n_D = int(V.shape[0] / 2)
    n_iD = n_D * 5


    n_m = n_D * (n_D -1)
    if (2*n_m + n_iD != n_r):
        print('Ocio! Reactions')
    
    # stoichiometric matrix, state vector, propensities
    V = np.asarray(V, dtype=np.int64)
    X = np.zeros((2 * n, 1), dtype=np.float64)
    a = np.zeros((n_r, 1), dtype=np.float64)

    # ---------------------------- First loop (over realizations) --------------------------- #
    for k in range(realisations):
        antimicrobial_added = False


        if k % 50 == 0:
            print('k=' + str(k))
        
        # Initialize population
        for i in range(n_D):
            X[2 * i] = js0
            X[2 * i + 1] = jr0
        
        # Initalize fitnesses and time
        t = 0
        fse = fs
        gse = gs

     
        ## ++++++++++++++++++ Second loop (over time) ++++++++++++++++++ ##
        while(t <= t_final):
          
            for y in range( n_D ):
                a[0 + y * 5] = fse * (1 - mi1) * (1 - (X[0 + y * 2] + X[1 + y * 2]) / K) * X[0 + y * 2]
                a[1 + y * 5] = gse * X[0 + y * 2]
                a[2 + y * 5] = fse * mi1 * (1 - (X[0 + y * 2] + X[1 + y * 2]) / K) * X[0 + y * 2]
                a[3 + y * 5] = fr * (1 - (X[0 + y * 2] + X[1 + y * 2]) / K) * X[1 + y * 2]
                a[4 + y * 5] = gr * X[1 + y * 2]
            for g in range(n_D):
                for h in range(n_D - 1):
                    a[n_iD + (n_D-1) * g + h] = gamma * X[2 * g]
            for g in range(n_D):
                for h in range(n_D - 1):
                    a[n_iD + n_m + (n_D-1) * g + h] = gamma * X[2 * g + 1]
            
            # If migrations are not rare, avoid going over the carrying capacity
            for j in range(n_D):
                if (X[0 + j * 2] + X[1 + j * 2] >= K):
                    a[0 + j * 5] = 0
                    a[3 + j * 5] = 0

            asum = np.sum(a)
            cumulative = np.cumsum(a) / asum

            xi = np.random.rand(2)
            tau = (np.log(1 / xi[1]) / asum)
            
            ### ......... Condition A: before Tadd and after Tadd with AM applied already ....... ###
            if (t + tau) < Tadd or ((t + tau) >= Tadd and antimicrobial_added):
                t = t + tau

                j = []
                for i in range(len(cumulative)):
                    if xi[0] < cumulative[i] and cumulative[i] != 0:
                        j.append(i)

                j_int = int(min(j))
                v_sel = V[:, j_int]
                v_sel_contig = np.ascontiguousarray(v_sel)

                new_shape = (n_D*2, 1)

                v_sel_contig = v_sel_contig.reshape(new_shape)
                X = X + v_sel_contig

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

                if is_close_to_final_time(t, t_final, 1e-3):
                    print("t is close to t_final within the specified precision.")
                    final_state.append(X.copy())
                    break
            
            ### ......... Condition B: application of AM at Tadd ...... #
            elif (t + tau) >= Tadd and not antimicrobial_added:
                print("tadd reached")
                t = Tadd
                state_beforeAM.append(X.copy())

                non_zero_exists = False
                for i in range(1, len(X), 2):
                    if X[i] != 0:
                        non_zero_exists = True
                        break

                if non_zero_exists:
                    pres_mut_bAM += 1

                fse = fsp
                gse = gsp
                print('Add antimicrobial')
                antimicrobial_added = True
                
        ## ++++++++++++++++++++++++++ End of second loop +++++++++++++++++++ ##
        
    # ------------------------ End of first loop --------------------------#

    return count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state, pres_mut_bAM, pres_mut_afterAM


#%% Parameters
# Build the V matrix for the structure and the number of demes of interest
V = build_Vmatrix_clique(5)
print(V)
fs = 1
gs = 0.1
fsp = 0

fr = 0.9
gsp = 0.1 
gr = 0.1 

gamma=  1e-4
mi1 = 1e-5

K = 200

js0 =  0.1 *K
jr0 = 0 

n = 5 # number of demes, spatial structure

Tadd = 4000000

realisations = 20 # Number of simulations (=runs)

t = 0
t_final = 100000000
EP = (1 - gs / fs) * K

#%%

# Run the simulation
count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state_end, pres_mut_bAM, pres_mut_aAM = simulate(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, gamma, js0, jr0, t_final, n)

            
#%%

print(sys.argv)
i = int(sys.argv[1])
print(i)

# Define the names to save the outputs
output_surv = f'survival_CLIQUE_fr{fr}_T{Tadd}_{i}.npy'
output_ext = f'extinction_CLIQUE_fr{fr}_T{Tadd}_{i}.npy'
output_state = f'state_CLIQUE_fr{fr}_beforeAM_T{Tadd}_{i}.npy'
final_state = f'state_CLIQUE_fr{fr}_final_T{Tadd}_{i}.npy'
count_fix_bAM = f'count_fix_bAM_fr{fr}_CLIQUE_T{Tadd}_{i}.npy'

np.save(output_state, state_beforeAM)
np.save(output_surv, fixations_R)
np.save(output_ext, extinctions_R)
np.save(final_state, final_state_end)
np.save(count_fix_bAM, count_fix_beforeAM)
