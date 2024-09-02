## DrugRes_StructPop
Repository for "Spatial structure facilitates evolutionary rescue by cost-free drug resistance" (2024) by C. Fruet, E. L. Müller, C. Loverdo, and A.-F. Bitbol.

## Requirements
In order to use the programs, NumPy and Numba are required. 

## Usage
The file `DrugRes_WMP.py` sets up and runs several Gillespie-based stochastic simulation of two bacterial types - one sensitive and one resistant to a drug - arranged in a well-mixed population. The drug is added at a user-defined timepoint that can be adjusted within the file. Each simulation continues until either the extinction or fixation of the resistant population occurs after the drug is added. The program saves the relevant data: survival and extinction times of resistant mutants, the state of the system before drug addition and the final state of the system. Additionally, since resistance could fix before drug addition when the drug addition time is longer than the average time it takes for a successful mutant to appear, a counter is also saved to track the number of times this occurs.

The files `DrugRes_clique.py`, `DrugRes_lattice.py` and `DrugRes_star.py` first set up migration matrices for two bacterial types organized in a structured population, with the spatial structure defined by a clique, lattice, or star graph, respectively. The programs then run Gillespie-based stochastic simulation similarly to the well-mixed case described above.

The saved files are then used for further analysis, as described in the Supplementary Information of "Spatial structure facilitates evolutionary rescue by cost-free drug resistance". In particular, the extinction probability is calculated as the length of the vector of extinction times, and the survival probability as its complement to 1.
