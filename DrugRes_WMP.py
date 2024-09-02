# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:42:36 2023

@author: fruet

Implementation of the Gillespie algorithm with the application of the antimicrobial (you get to decide at the beginning, where
                                                                                     you decide the parameters).
This is for ONE-SHOT application (i.e., apply antimicrobial only once and leave it on, no switches).

It is numba-accelerated for much faster calculations! :)

Aim: run several realizations and calculate the probability of survival in the case of a well-mixed population.
All cells that start with "WELL-MIXED" implement a Gillespie simululation for "realisations" runs,
with K=1000 and initial population js0=100.
The variables fixations_R and extinctions_R contain the times at which fixation and extinction of R-types
happen, respectively. Note: their lengths should sum up to the total simulation number, otherwise you have to
increase the final time. If you divide their lengths by the number of runs, you will get the probability
of survival or extinction of R-types, respectively.
If you run the cell after the simulation, you will be given on terminal the number of systems that
get extinct, the extinction probability, and the error as the standard error of the proportion.

NOTE! If you use the first part to calculate the single-deme average, you must change K and js0!

"""

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

#%%

@njit
def build_V_matrix(nspecies):
    V = np.array([[1, 0, -1, 0,  0],
                 [0, 1,  0, 1, -1]])
    
    return V


#%%

@njit
def simulate_gillespie_WM(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, js0, jr0, t_final, n):
    extinctions_R = []
    fixations_R = []
    state_beforeAM = []
    final_state = []
    ks = []
    count_fix_beforeAM = 0
    V = np.asarray(V, dtype=np.int64)
    X = np.zeros((2 * n, 1), dtype=np.float64)
    a = np.zeros((5, 1), dtype=np.float64)

    for k in range(realisations):
        antimicrobial_added = False

        if (k % 50 == 0):
            print('k=' + str(k))

        X[0] = js0
        X[1] = jr0

        t = 0

        fse = fs
        gse = gs

        while t < t_final:
            a[0] = fse * (1 - mi1) * (1 - (X[0] + X[1]) / K) * X[0]
            a[1] = fse * mi1 * (1 - (X[0] + X[1]) / K) * X[0]
            a[2] = gse * X[0]
            a[3] = fr * (1 - (X[0] + X[1]) / K) * X[1]
            a[4] = gr * X[1]
            asum = np.sum(a)
            cumulative = np.cumsum(a) / asum

            xi = np.random.rand(2)

            tau = (np.log(1 / xi[1]) / asum)

            if (t + tau) < Tadd or ((t + tau) >= Tadd and antimicrobial_added):
                t = t + tau
                j = []
                for i in range(len(cumulative)):
                    if xi[0] < cumulative[i] and cumulative[i] != 0:
                        j.append(i)

                j_int = int(min(j))

                v_sel = V[:, j_int]
                v_sel_contig = np.ascontiguousarray(v_sel)

                new_shape = (2, 1)

                v_sel_contig = v_sel_contig.reshape(new_shape)
                X = X + v_sel_contig


                if X[1] == 0 and X[0] == 0:
                    extinctions_R.append(t)
                    final_state.append(X.copy())
                    print("extinction of R")
                    ks.append(k)
                    if len(final_state) != len(state_beforeAM):
                        state_beforeAM.append(X.copy())
                    break

                if X[1] == EP and X[0] == 0:
                    fixations_R.append(t)
                    final_state.append(X.copy())
                    print("fixation of R")
                    ks.append(k)
                    if len(final_state) != len(state_beforeAM):
                        print('pop fix before AM')
                        state_beforeAM.append(X.copy())
                        count_fix_beforeAM += 1
                    break

                if is_close_to_final_time(t, t_final, 1e-3):
                    print("t is close to t_final within the specified precision.")
                    final_state.append(X.copy())
                    break

            elif (t + tau) >= Tadd and not antimicrobial_added:
                state_beforeAM.append(X.copy())
                t = Tadd
                fse = fsp
                gse = gsp
                antimicrobial_added = True

    return ks, count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state




#%% 

# Relative fitnesses and death rates for S-types and R-types
fs = 1
gs = 0.1
fr = 1
gr = 0.1 

V = build_V_matrix(2)

# Decide which kind of antimicrobial
Biostatic = True
Biocidal = False

if Biostatic == True:
    fsp = 0
    gsp = 0.1 
elif Biocidal == True:
    fsp = 1
    gsp = 1
else:
    print('Ocio! Error!')

# Mutation rate
mi1 = 1e-5

# Carrying capacity
K = 1000

# Initial condition
js0 = 100
jr0 = 0 
n = 1
EP = K * (1 - gs / fs)

# Final time and time for periodic switching
t_final = 10000000


# Addition time of AM
Tadd = 4000000

realisations = 100 #1000 # Number of simulations (=runs)

#%%

ks, count_fix_beforeAM, extinctions_R, fixations_R, state_beforeAM, final_state_end = simulate_gillespie_WM(V, realisations, Tadd, fs, gs, fsp, gsp, EP, K, mi1, fr, gr, js0, jr0, t_final, n)


#%%

print(sys.argv)
i=int(sys.argv[1])
print(i)

output_surv = f'survivalR_WM_T{Tadd}_{i}.npy'
output_ext = f'extinctionR_WM_T{Tadd}_{i}.npy'
output_state = f'state_WM_beforeAM_T{Tadd}_{i}.npy'
final_state = f'state_WM_final_T{Tadd}_{i}.npy'
count_fix_bAM = f'count_fix_bAM_WM_T{Tadd}_{i}.npy'

np.save(output_state, state_beforeAM)
np.save(output_surv, fixations_R)
np.save(output_ext, extinctions_R)
np.save(final_state, final_state_end)
np.save(count_fix_bAM, count_fix_beforeAM)

