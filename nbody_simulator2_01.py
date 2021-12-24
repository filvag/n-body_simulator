#!/usr/bin/env python3

import argparse #A library that allows parsing arguments from the command line
import matplotlib as mpl ## to overcome overflowError in draw_path
import matplotlib.pyplot as plt #A library that can be used to make graphs
import numpy as np #The core library for scientific computing in Python. 
import timeit
import csv
#numpy provides a high-performance multidimensional array object, and tools for working with these arrays. 
#numpy data structures perform better in memory, performance and fucntionalities.
mpl.rcParams['agg.path.chunksize'] = 10000 ## to overcome overflowError in draw_path

###LEAFPROG INTEGRATOR###
def Leapfrog_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
    print("Leapfrog Integrator")
    dxdt0  = first_order_n_body(x, masses, nbodies, G) 
    for count,t in enumerate(time_vector[1:],1):
        x[0:3*nbodies] = x[0:3*nbodies] + x[3*nbodies:]*dt+0.5 * dxdt0[3*nbodies:] *dt**2
        dxdt = first_order_n_body(x, masses, nbodies, G)
        x[3*nbodies:] = x[3*nbodies:] + 0.5 * (dxdt0[3*nbodies:] + dxdt[3*nbodies:]) * dt
        state_vector[count,:] = x
        dxdt0 = dxdt
    return state_vector

### ADAMS BASHFORTH INTEGRATOR ###
def Adams_Bashforth_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
    print("Adam-Bashforth Integrator")
    dxdt0  = first_order_n_body(x, masses,nbodies, G) 
    x = x + dxdt0 * dt
    state_vector[1,:] = x
    for count,t in enumerate(time_vector[1:],1):
        dxdt = first_order_n_body(x, masses, nbodies, G)
        x = x + 0.5 * dt * (3 * dxdt - dxdt0)
        state_vector[count,:] = x
        dxdt0 = dxdt
        x0 = x
    return state_vector

###RUNGE KUTTA INTEGRATION###
##A single-step method, explicit
def Runge_Kutta_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
    print("Runge-Kutta Integrator")
    for count,t in enumerate(time_vector[1:],1):
        k1 = first_order_n_body(x, masses, nbodies, G) 
        k2 = first_order_n_body(x+0.5*dt*k1, masses, nbodies, G)
        k3 = first_order_n_body(x+0.5*dt*k2, masses, nbodies, G)
        k4 = first_order_n_body(x+dt*k3, masses, nbodies, G)
        x = x + dt * (k1+2*k2+2*k3+k4)/6.0
        state_vector[count,:] = x
    return state_vector 

###FORWARD EULER INTEGRATION###
###Every step is advanced by assuming that the object drifts at a uniform velocity during a small time interval dt
def forward_Euler_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
    print("Euler Integrator")
    for count,t in enumerate(time_vector[1:],1):
        dxdt = first_order_n_body(x, masses, nbodies, G)
        x = x + dxdt * dt #advance step
        state_vector[count,:] = x
    return state_vector 

def first_order_n_body(x, masses, nbodies, G):
    dxdt = x * 0.0 #Initialise vector
    for j in range(0, nbodies):
        Rj = x[j*3:(j+1)*3]
        #velocities
        dxdt[j*3:(j+1)*3] = x[(nbodies+j)*3:(nbodies+j+1)*3]
        #accelerations
        aj = Rj*0
        for k in range(0, nbodies):
            if (j==k): #skip equal indices
                continue
            if (masses[k]==0):#if a body has no mass, skip it
                continue
            Rk = x[k*3:(k+1)*3]
            r = Rj-Rk
            aj += -G*masses[k]*r/np.linalg.norm(r)**3
        dxdt[(nbodies+j)*3:(nbodies+j+1)*3] = aj
    return dxdt

#####################################################################################################
### WISDOM HOLMAN INTEGRATOR ###
def Wisdom_Holman_integration(t_end, dt, masses, x, nbodies, G, skip_step):
    print("Wisdom-Holman Integrator")

    # Initialise array buffer to store solution
    npts = int(np.floor(t_end/(dt * skip_step))) + 1
    #state_vector = np.zeros((npts, nbodies * 6))
    state_vector = np.zeros((npts,len(x)))
    time_vector = np.zeros(npts)
    #time_vector = np.linspace(0,dt*(npts-1), npts)#vector of times

    cart = np.zeros(nbodies * 6) #Allocate cartesian array
    cart[0:] = x[0:] # Intialise it
    
    # Store initial solution
    state_vector[0, :] = cart
    time_vector[0] = 0 

    # Compute etas (interior masses)
    eta = np.zeros(nbodies)
    eta[0] = masses[0]
    for body in range(1, nbodies):
        eta[body] = masses[body] + eta[body - 1]

    jacobi = cart2jacobi(cart, masses, nbodies, eta)# compute jacobi coordinates and initial accelleration
    accel = compute_acceleration(cart, jacobi, masses, nbodies, G, eta)
    istep=0
    isol=1
    t = 0
    drift1 = True
    extend2 = True
    while t < t_end:
        jacobi, accel = wh_advance_step(jacobi, t, dt, masses, nbodies, accel, G, eta)#advance one step
        t += dt #advance time
        istep += 1 #advance step counter
        #store the solution
        if (istep % skip_step ==0):
            cart = jacobi2cart(jacobi, masses, nbodies, eta) #convert to cartesian
            state_vector[isol, :] = cart
            time_vector[isol] = t
            isol += 1
    return state_vector[0: isol, :]

##Advance one step using Wisdom-Holman integrator
def wh_advance_step(x,t,dt,masses,nbodies,accel,G,eta):
    jacobi = np.zeros(nbodies * 6)
    jacobi = x
    jacobi = wh_kick(jacobi, 0.5*dt, nbodies, accel)#kick for dt/2 using the acceleration at t 
    jacobi = wh_drift(jacobi, dt, masses, nbodies, G, eta) #drift for dt
    cart = jacobi2cart(jacobi, masses, nbodies, eta)#covert from jacobi to cartesian for computing the acceleration
    accel = compute_acceleration(cart, jacobi, masses, nbodies, G, eta)#compute acceleration at t+dt
    jacobi = wh_kick(jacobi, 0.5*dt,nbodies, accel)#kick for dt/2 using the acceleration at t+dt
    return jacobi,accel

##Comnpute the velocity kick
def wh_kick(x,dt, nbodies, accel):
    kick = np.zeros(nbodies*6)#allocate output array
    kick[0:]=x[0:]#initialise it
    kick[nbodies*3:] += accel*dt #apply kick (for velocities only)
    return kick

##Drift the state of all bodies (Keplerian propagation)
def wh_drift(x, dt, masses, nbodies, G, eta):
    drift = np.zeros(nbodies*6) #drifted state
    #propagate each body assuming Keplerian motion
    for body in range(1,nbodies):
        gm = masses[0]*eta[body]/eta[body-1]*G #compute equivalent GM
        pos0 = x[body*3: (body+1)*3]#initial position
        vel0 = x[(nbodies + body)*3: (nbodies + body + 1)*3]#initial velocity
        pos, vel = propagate_kepler(0.0, dt, pos0, vel0, gm) #propagation
        drift[body*3: (body+1)*3] = pos
        drift[(nbodies+body)*3: (nbodies+body+1)*3] = vel
    return drift

## Evaluate the first four Stumpff functions
def stumpff_functions(z):
    n = 0
    while (abs(z) > 0.1):
        n +=1 
        x /= 4.0
    c3 = (1.-z*(1.-z*(1.-z*(1.-z*(1.-z*(1.-z/210.) / 156.) / 110.) / 72.) /42.) / 20.) / 6.
    c2 = (1.-z*(1.-z*(1.-z*(1.-z*(1.-z*(1.-z/182.) / 132.) / 90.) / 56.) /30.) / 12.) / 2.
    c1 = 1.0 - z * c3
    c0 = 1.0 - z * c2
    while (n>0):
        n -= 1
        c3 = (c2 + c0 * c3) / 4.
        c2 = c1 * c1 / 2.
        c1 = c0 * c1
        c0 = 2. * c0 * c0 - 1.
    return c0, c1, c2, c3

## Analytic propagation of Kepler's problem in universal variable 
def propagate_kepler(t0, tf, vr0, vv0, gm):
    #check for trivial propagation
    if (t0==tf):
        vrf = vr0
        vvf = vv0
        return
    dt = tf-t0 #compute time step
    tol = 1e-12 #internal tolerance for so;ving Kepler's equation
    r0 = np.linalg.norm(vr0) #compute the magnitude of the initial position vector
    v0 = np.linalg.norm(vv0) #compute the magnitude of the initial velocity vector
    alpha = -(v0**2 - 2.0 *gm / r0) #alpha parameter
    dr0 = np.dot(vr0, vv0)/r0 #radial velocity
    #Solve Kepler's equation
    s = dt/r0
    for j in range(0,50):
        c0, c1, c2, c3 = stumpff_functions(alpha * s**2) #compute Stumpff function
        F = r0 * s * c1 + r0* dr0 * s**2 * c2 + gm *s**3 *c3 - dt #Evaluate Kepler's equation
        if (abs(F) < tol) and ('vr' in vars() or 'vr' in globals()) and ('vv' in vars() or 'vv' in globals()): #convergence, check function value
            break
        dF = r0 * c0 + r0 * dr0 * s * c1 + gm * s**2 * c2 #compute derivative
        ds = -F / dF #Find step size
        if (abs(ds) < tol) and ('vr' in vars() or 'vr' in globals()) and ('vv' in vars() or 'vv' in globals()): #convergence, check size of next step
            break
        s += ds #advance step
        r = dF #the radial distance is equal to the derivative of F
        f = 1.0 - gm * s**2 * c2 / r0 
        g = dt - gm * s**3 * c3
        df = -gm / (r*r0) * s * c1
        dg = 1.0 - gm / r * s**2 * c2
        vr = f * vr0 + g * vv0 #compute position vector
        vv = df * vr0 + dg * vv0 #compute velocity vector
    return vr, vv

##Compute pertubiing acceleration in Jacobi coordinates
def compute_acceleration(cart, jacobi, masses, nbodies, G, eta):
    accel = np.zeros(nbodies * 3) #Allocate output array
    aux = np.zeros(3) # Will be used for intermediate calculations
    for i in range(1, nbodies):
        r0i = cart[i*3: (i+1)*3]-cart[0:3]
        accel[i*3:(i+1)*3]=masses[0]*eta[i]/eta[i-1]*(jacobi[i*3:(i+1)*3]/np.linalg.norm(jacobi[i*3:(i+1)*3])**3-r0i/np.linalg.norm(r0i)**3)
        aux *= 0.0#initialisations
        for j in range(1, i):
            rji = cart[i*3:(i+1)*3]-cart[j*3:(j+1)*3]
            aux += masses[i]*rji/np.linalg.norm(rji)**3
        accel[i*3:(i+1)*3] += -eta[i]/eta[i-1]*aux
        aux *= 0.0#re- initialise for next calculations
        for j in range(0,i):
            for k in range(i+1, nbodies):
                rjk = cart[k*3:(k+1)*3]-cart[j*3:(j+1)*3]
                aux += masses[j]*masses[k]*rjk/np.linalg.norm(rjk)**3
        accel[i*3:(i+1)*3] += -aux/eta[i-1]
    accel *= G
    return accel

## Convert from Cartesian to Jacobi coordinates
def cart2jacobi(x, masses, nbodies, eta):
    jacobi = np.zeros(nbodies * 6) #Allocate output
    R = masses[0] * x[0:3]
    V = masses[0] * x[nbodies *3: (nbodies + 1) * 3]
    for i in range(1, nbodies): 
        jacobi[i * 3: (i+1)*3] = x[i * 3: (i+1) * 3] - R / eta[i - 1] 
        jacobi[(nbodies+i)*3: (nbodies+i+1)*3]=x[(nbodies+i)*3:(nbodies+i+1)*3]-V/eta[i-1] 
        R = R*(1+masses[i]/eta[i-1])+masses[i]*jacobi[i*3:(i+1)*3]
        V = V*(1+masses[i]/eta[i-1])+masses[i]*jacobi[(nbodies+i)*3:(nbodies+i+1)*3]
    jacobi[0: 3] = R /eta[-1]
    jacobi[nbodies*3: (nbodies+1)*3] = V/eta[-1]
    return jacobi

## Convert from Jacobi to Cartesian coordinates
def jacobi2cart(x, masses, nbodies, eta):
    cart = np.zeros(nbodies*6) #allocate output
    R = x[0: 3] * eta[-1]
    V = x[nbodies*3: (nbodies+1)*3]*eta[-1]
    for i in range(nbodies-1, 0, -1):
        R = (R-masses[i]*x[i*3: (i+1)*3])/eta[i]
        V = (V-masses[i]*x[(nbodies+i)*3: (nbodies+i+1)*3])/eta[i]
        cart[i*3: (i+1)*3] = x[i*3: (i+1)*3]+R
        cart[(nbodies+i)*3: (nbodies+i+1)*3]=x[(nbodies+i)*3: (nbodies+i+1)*3]+V
        R = eta[i-1]*R
        V = eta[i-1]*V
    cart[0: 3] = R / masses[0]
    cart[nbodies*3: (nbodies+1)*3] = V/masses[0]
    return cart
###################################################################################################################


def calculate_energy_angmomentum_eccentricity(state_vector, masses, energy_on, angmomentum_on,eccentricity_body, nbodies, G):
    energy = state_vector[:,0] * 0
    angmomentum = state_vector[:,0] * 0
    eccentricity = state_vector[:,0] * 0
    print("calculating energy, angular momentum, eccentricity, or a combination")
    for count,x in enumerate(state_vector):
        if energy_on:
            energy[count]=0
            for i in range(0, nbodies):
                energy[count] += 0.5 * masses[i] * np.linalg.norm(x[(nbodies+i)*3:(nbodies+1+i)*3])**2
                for j in range(0,nbodies):
                    if (i==j): continue
                    energy[count] -= 0.5*G*masses[i]*masses[j]/np.linalg.norm(x[i*3:(i+1)*3]-x[(j*3):(j+1)*3])
        if angmomentum_on:
            angmomentum[count]=0
            for i in range(0,nbodies):
                angmomentum[count]=np.linalg.norm(angmomentum[count]+masses[i]*np.cross(x[i*3:i*3+3],x[(nbodies+i)*3:(nbodies+i)*3+3]))
        if eccentricity_body:
            r1 = x[0:3] -x[eccentricity_body*3:eccentricity_body*3+3]
        #    v1 = y[6:9] -y[9:12]
            v1 = x[nbodies:(nbodies+3)] -x[(nbodies+eccentricity_body)*3:(nbodies+eccentricity_body)*3+3]
        #    vecc = np.cross(v1, np.cross(r1,v1)) / (G * (M1+M2)) -r1 / np.linalg.norm(r1)
            vecc = np.cross(v1, np.cross(r1,v1)) / (G * (masses[0]+masses[eccentricity_body])) -r1 / np.linalg.norm(r1)
            eccentricity[count] = np.linalg.norm(vecc)
    return energy,angmomentum,eccentricity 

def main():
    #parsing the arguments first
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', dest='filename', help='Input file name (for bodies initial state)',required=True)
    parser.add_argument('-E', '--Euler', action="store_true", default=False, dest='euler', help='Use Euler Integration')  
    parser.add_argument('-R', '--Runge_Kutta', action="store_true", default=False, dest='runge_kutta', help='Use Runke Kutta Integration')  
    parser.add_argument('-A', '--Adams_Bashforth', action="store_true", default=False, dest='adams_bashforth', help='Use Adams Bashforth Integration')  
    parser.add_argument('-L', '--Leapfrog', action="store_true", default=True, dest='leafprog', help='Use Leapfrog Integration (default)')  
    parser.add_argument('-W', '--Wisdom_Holman', action="store_true", default=False, dest='wisdom_holman', help='Use Wisdom-Holman Integration')  
    parser.add_argument('-e', '--energy', action="store_true", default=False, dest='energy', help='plot Energy (default: False)')  
    parser.add_argument('-a', '--angular_momentum', action="store_true", default=False, dest='angmomentum', help='plot Angular Momentum (default: False)')  
    parser.add_argument('-ecc', '--eccentricity', action="store", type=int, default=0, dest='eccentricity', help='plot Eccentricity for a body other than body "0" (default: False)')  
    parser.add_argument('-d', '--dt', type=float, dest='dt', help='Integration time step (in earth days)', default=1.0)  
    parser.add_argument('-t', '--t_end', type=float, dest='t_end', help='Termination time (earth years)', default=1)  
    parser.add_argument('-g', '--gravitational_constant', type=float, dest='gconstant', help='Universal Gravitational constant (AU^3 * days-2 * solar_mass^-1)', default=0.00029591220828559)  
    parser.add_argument('-s', '--solar', action="store_true", default=False, dest='solar', help='compute centre of mass (for the solar system)')  
    args = parser.parse_args()
   
    t_end = 365.25*args.t_end #convert years to days
    G = float(args.gconstant)
    body_names =[]
    bodies = []
    masses =[]
    try:
        with open(args.filename) as file_object:
            reader = csv.reader(file_object)
            header_row = next(reader)#read the header of the cvs; not used but we move to next row
            #print(header_row)
            for row in reader:
                if row[0][:1]!='#':
                    bodies.append(row)
    except FileNotFoundError:
        print(f"Could not find file {filename}")
        exit(0)

    nbodies = len(bodies)
    if args.eccentricity > nbodies-1 or args.eccentricity < 0:
        print("the body to estimate the eccentricity should be greater than or equal to 1 AND smaller than the total number of bodies", nbodies)
        exit(0)
    x0 = np.zeros(nbodies *6)

    for count,b in enumerate(bodies):
        masses.append(float(b[6]))
        x0[count*3]=float(b[0])
        x0[count*3+1]=float(b[1])
        x0[count*3+2]=float(b[2])
        x0[(count+nbodies)*3]=float(b[3])
        x0[(count+nbodies)*3+1]=float(b[4])
        x0[(count+nbodies)*3+2]=float(b[5])
        body_names.append(b[7])

    if args.solar:#fixes position/velocity of body [0] (assuming it is the bigest of teh system) ==> to have a better estimate based on real centre of mass of the system (optional)
        for i in range(1,nbodies):
            x0[0:3] -= masses[i] * x0 [i*3:(i+1)*3]
            x0[nbodies*3:(nbodies+1)*3] -= masses[i]*x0[(nbodies+i)*3:(nbodies+i+1)*3]

    #intialise space and time vectors
    if not args.wisdom_holman: ##Wisdom Holman implements its initialisation due to step_time
        npts = int(np.floor((t_end/args.dt)+1)) #number of points in the dense output
        time_vector = np.linspace(0,args.dt*(npts-1), npts)#vector of times
        state_vector = np.zeros((npts,len(x0)))
        state_vector[0,:] = x0 

    ###INTEGRATION###
    integrator = ""
    start = timeit.default_timer()
    if args.euler:
        state_vector = forward_Euler_integration(t_end,args.dt,masses,x0,state_vector,time_vector,nbodies,G)
        integrator = "Euler integrator"
    elif args.runge_kutta:
        state_vector = Runge_Kutta_integration(t_end,args.dt,masses,x0,state_vector,time_vector,nbodies,G)
        integrator = "Runge Kutta integrator"
    elif args.adams_bashforth:
        state_vector = Adams_Bashforth_integration(t_end,args.dt,masses,x0,state_vector,time_vector,nbodies,G)
        integrator = "Adams Bashforth integrator"
    elif args.wisdom_holman:
        state_vector = Wisdom_Holman_integration(t_end,args.dt,masses,x0,nbodies,G,1000)
        integrator = "Wisdom Holman integrator"
    elif args.leafprog:
        state_vector = Leapfrog_integration(t_end,args.dt,masses,x0,state_vector,time_vector,nbodies,G)
        integrator = "Leapfrog integrator"
    else:
        print("Please define an Integrator to use")
        exit(0)
    stop = timeit.default_timer()
    runtime = stop-start
    print("runtime =",runtime,"sec")

    ###PLOTTING RESULTS###
    # make the plot when finish with all calculations
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for ibody in range (0,nbodies):
        #traj = ax.plot(state_vector[:,ibody*3], state_vector[:,1+ibody*3])  
        #ax.plot(state_vector[0, ibody*3], state_vector[0,1+ibody*3], "o",color=traj[0].get_color(), label=body_names[ibody])
        traj = ax.plot(state_vector[:,ibody*3], state_vector[:,2+ibody*3])  
        ax.plot(state_vector[0, ibody*3], state_vector[0,2+ibody*3], "o",color=traj[0].get_color(), label=body_names[ibody])
    ax.set_xlabel('X (Astronomical Units)')
    ax.set_ylabel('Y (Astronomical Units)')
    plt.legend(loc='lower left')
    #plt.legend()
    plt.title("runtime="+str(round(runtime,2))+" sec, t="+str(args.t_end)+" y, dt="+str(args.dt)+" y")
    plt.suptitle(integrator, weight = 'bold')
    fig.savefig("trajectory.png")
    if args.energy or args.angmomentum or args.eccentricity:
        energy,angmomentum,eccentricity = calculate_energy_angmomentum_eccentricity(state_vector, masses, args.energy,args.angmomentum,args.eccentricity,nbodies,G)
        if args.energy:
            fig2=plt.figure(2)
            if args.adams_bashforth:
                plt.semilogy(time_vector, np.abs((energy-energy[1])/energy[1]))
                plt.ylabel('|(E(t)-E1)/E1|')
            else:
                plt.semilogy(time_vector, np.abs((energy-energy[0])/energy[0]))
                plt.ylabel('|(E(t)-E0)/E0|')
            plt.xlabel('Time (earth days)')
            plt.title("runtime="+str(round(runtime,2))+" sec, t="+str(args.t_end)+" y, dt="+str(args.dt)+" y")
            plt.suptitle(integrator, weight = 'bold')
            fig2.savefig("energy.png")
        if args.eccentricity:
            fig4=plt.figure(4)
            plt.semilogy(time_vector, np.abs((eccentricity-eccentricity[0])/eccentricity[0]))
            plt.xlabel('Time (earth days)')
            plt.ylabel('eccentricity '+body_names[args.eccentricity])
            plt.title("runtime="+str(round(runtime,2))+" sec, t="+str(args.t_end)+" y, dt="+str(args.dt)+" y")
            plt.suptitle(integrator, weight = 'bold')
            fig4.savefig("eccentricity.png")
            #for k in range(1,9):
            #    plt.plot([4 * np.pi * k, 4 * np.pi * k], [1e-10,1e-2], "k:")
        if args.angmomentum:
            fig3=plt.figure(3)
            plt.semilogy(time_vector, np.abs((angmomentum-angmomentum[0])/angmomentum[0]))
            plt.xlabel('Time (earth days)')
            plt.ylabel('angmomentum')
            plt.title("runtime="+str(round(runtime,2))+" sec, t="+str(args.t_end)+" y, dt="+str(args.dt)+" y")
            plt.suptitle(integrator, weight = 'bold')
            fig3.savefig("angmomentum.png")
    plt.show()

if __name__ == '__main__':
    main()
