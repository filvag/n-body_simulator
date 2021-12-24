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
_-i_  <filename> The input file name (for bodies initial state)

_-E_ Use the Euler integrator  
_-R_ Use Runke Kutta Integration  
_-A_ Use Adams Bashforth Integration  
_-L_ Use Leapfrog Integration (default) 
 
_-e_ plot Energy (default: False)  
_-a_ plot Angular Momentum (default: False)  
_-ecc_ plot Eccentricity for a body other than body "0" (default: False)  
_-d <time step>_ The integration time step (in earth days)', default=1.0  
_-t <time>_ Termination time (earth years)', default=1)  
_-g <gravitational_constant>_ The Universal Gravitational constant (AU^3 * days-2 * solar_mass^-1)', default=0.00029591220828559  
_-s <solar>_  compute centre of mass (for the solar system)  
