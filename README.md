# n-body_simulator
An n-body simulator to study planetary and exoplanetary systems

_nbody_simulator_01.py_ implements four fixed-step size integrators: 

a. Forward Euler
b. Leapfrog
c. Adams Bashforth
d. Runge Kutta

_nbody_simulator2_01.py_ implements the above plus the Wisdom Holman

The csv files are examples of files that cab be used as an input for the studies (our solar system, Keppler-11, etc.). 

The program plots the trajectories, as well as the system energy, angular momentum and eccentricity.

Credits to the "Moving Planets Around - An Introduction to N-Body Simulations Applied to Exoplanetary Systems" by Javier Roa, Adrian S. Hamers, Maxwell X. Cai and Nathan W. C. Leigh from MIT Press (https://mitpress.mit.edu/books/moving-planets-around) which was the source for these integrators. 

## How to use it
-i  <filename> The input file name (for bodies initial state)
-E Use the Euler integrator  
-R Use Runke Kutta Integration  
-A Use Adams Bashforth Integration  
-L Use Leapfrog Integration (default)  
-e plot Energy (default: False)')  
-a plot Angular Momentum (default: False)')  
-ecc plot Eccentricity for a body other than body "0" (default: False)')  
-d <time step> The integration time step (in earth days)', default=1.0  
-t <time> Termination time (earth years)', default=1)  
-g <gravitational_constant> The Universal Gravitational constant (AU^3 * days-2 * solar_mass^-1)', default=0.00029591220828559  
-s <solar> 'compute centre of mass (for the solar system)  
