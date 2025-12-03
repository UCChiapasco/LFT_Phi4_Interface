A Scalar Lattice Field Theory simulation developed as part of my Master's thesis. The simulation is a non-equilibrium Markov chain Monte Carlo
 simulation, designed mainly to compute the values of interface free energies, whose behavior can be compared to Effective String
 - LFT_Phi4.py contains two classes used for these simulations, as well as a get_works() function that returns the works performed during
   multiple non-equilibrium trajectory. These works can be used to evaluate the interface free energy through Jarzynski equality.
 - analisi.py contains a few useful functions based on the pyerrors library
 - main_scaling.py is an example main code used for the simulations. It stores the results in a .txt file
 - luscher_analysis.ipynb contains a few results obtained by processing main_scaling.py (or an analogue) on the Leonardo supercomputer at Cineca
 - scaling_analysis.ipynb contains results on the scaling of the algorithm obtained through the evaluation of the Kullback-Leibler divergence
   and Effective Sample Size
