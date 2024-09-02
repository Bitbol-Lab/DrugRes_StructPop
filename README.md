## DrugRes_StructPop
Repository for "Spatial structure facilitates evolutionary rescue by cost-free drug resistance" (2024) by C. Fruet, E. L. Müller, C. Loverdo, and A.-F. Bitbol.

## Requirements
In order to use the programs, NumPy and Numba are required. 

## Usage
The files run several Gillespie-based stochastic simulations of two bacterial strains, sensitive and resistant to a drug, arranged in a spatial structure (clique, lattice, star) or in a well-mixed population. The drug is added at a given timepoint and affects only sensitive individuals. The simulation is continued until extinction or fixation of the population.
