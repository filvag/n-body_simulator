#!/usr/bin/env python3

import argparse #A library that allows parsing arguments from the command line
import matplotlib as mpl ## to overcome overflowError in draw_path
import matplotlib.pyplot as plt #A library that can be used to make graphs
import numpy as np #The core library for scientific computing in Python. 
#numpy provides a high-performance multidimensional array object, and tools for working with these arrays. 
#numpy data structures perform better in memory, performance and fucntionalities.
import timeit
import csv
mpl.rcParams['agg.path.chunksize'] = 10000 ## to overcome overflowError in draw_path

def Leapfrog_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
    dxdt0  = first_order_n_body(x, masses, nbodies, G) 
    for count,t in enumerate(time_vector[1:],1):
        x[0:3*nbodies] = x[0:3*nbodies] + x[3*nbodies:]*dt+0.5 * dxdt0[3*nbodies:] *dt**2
        dxdt = first_order_n_body(x, masses, nbodies, G)
        x[3*nbodies:] = x[3*nbodies:] + 0.5 * (dxdt0[3*nbodies:] + dxdt[3*nbodies:]) * dt
        state_vector[count,:] = x
        dxdt0 = dxdt
    return state_vector

def Adams_Bashforth_integration(t_end, dt, masses, x, state_vector, time_vector, nbodies, G):
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
            v1 = x[nbodies:(nbodies+3)] -x[(nbodies+eccentricity_body)*3:(nbodies+eccentricity_body)*3+3]
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
        x0[count*3]=float(b[0])
        x0[count*3+1]=float(b[1])
        x0[count*3+2]=float(b[2])
        x0[(count+nbodies)*3]=float(b[3])
        x0[(count+nbodies)*3+1]=float(b[4])
        x0[(count+nbodies)*3+2]=float(b[5])
        masses.append(float(b[6]))
        body_names.append(b[7])

    #fixes position/velocity of body [0] (assuming it is the bigest of the system)
    #==> to have a better estimate based on real centre of mass of the system (optional)
    if args.solar: 
        for i in range(1,nbodies):
            x0[0:3] -= masses[i] * x0 [i*3:(i+1)*3]
            x0[nbodies*3:(nbodies+1)*3] -= masses[i]*x0[(nbodies+i)*3:(nbodies+i+1)*3]

    #initialise space and time vectors
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
        traj = ax.plot(state_vector[:,ibody*3], state_vector[:,2+ibody*3])  
        #ax.plot(state_vector[0, ibody*3], state_vector[0,1+ibody*3], "o",color=traj[0].get_color(), label=body_names[ibody])
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
        if args.angmomentum:
            fig3=plt.figure(3)
            plt.semilogy(time_vector, np.abs((angmomentum-angmomentum[0])/angmomentum[0]))
            plt.xlabel('Time (earth days)')
            plt.ylabel('angmomentum')
            plt.title("runtime="+str(round(runtime,2))+" sec, t="+str(args.t_end)+" y, dt="+str(args.dt)+" y")
            plt.suptitle(integrator, weight = 'bold')
            fig3.savefig("angmomentum.png")
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
    plt.show()

if __name__ == '__main__':
    main()
