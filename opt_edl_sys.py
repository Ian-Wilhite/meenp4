#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:58:03 2021

@author: Marvin Engineering Design Team
"""

import numpy as np
from subfunctions_Phase4 import *
from define_experiment import *
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
import pickle
import sys
import os
import csv
import random

print(f'1')

# the following calls instantiate the needed structs and also make some of
# our design selections (battery type, etc.)
planet = define_planet()
edl_system = define_edl_system()
mission_events = define_mission_events()
edl_system = define_chassis(edl_system,'carbon')
edl_system = define_motor(edl_system,'base')
edl_system = define_batt_pack(edl_system,'PbAcid-1', 10)
tmax = 5000

# Overrides what might be in the loaded data to establish our desired
# initial conditions
edl_system['altitude'] = 11000    # [m] initial altitude
edl_system['velocity'] = -587     # [m/s] initial velocity
edl_system['parachute']['deployed'] = True   # our parachute is open
edl_system['parachute']['ejected'] = False   # and still attached
edl_system['rover']['on_ground'] = False # the rover has not yet landed

experiment, end_event = experiment1()

# constraints
max_rover_velocity = -1  # this is during the landing phase
min_strength=40000
max_cost = 7.2e6
max_batt_energy_per_meter = edl_system['rover']['power_subsys']['battery']['capacity']/1000

print(f'2')

# ******************************
# DEFINING THE OPTIMIZATION PROBLEM
# ****
# Design vector elements (in order):
#   - parachute diameter [m]
#   - wheel radius [m]
#   - chassis mass [kg]
#   - speed reducer gear diameter (d2) [m]
#   - rocket fuel mass [kg]
#

# search bounds
#x_lb = np.array([14, 0.2, 250, 0.05, 100])
#x_ub = np.array([19, 0.7, 800, 0.12, 290])
bounds = Bounds([14, 0.2, 250, 0.05, 100], [19, 0.7, 800, 0.12, 290])
xbest = np.array([14.0, 0.4, 250.0, 0.12, 100.0]) 

# lambda for the objective function
obj_f = lambda x: obj_fun_time(x,edl_system,planet,mission_events,tmax,
                               experiment,end_event)

# lambda for the constraint functions
#   ineq_cons is for SLSQP
#   nonlinear_constraint is for trust-constr
cons_f = lambda x: constraints_edl_system(x,edl_system,planet,mission_events,
                                          tmax,experiment,end_event,min_strength,
                                          max_rover_velocity,max_cost,max_batt_energy_per_meter)

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 0)  # for trust-constr
ineq_cons = {'type' : 'ineq',
             'fun' : lambda x: -1*constraints_edl_system(x,edl_system,planet,
                                                         mission_events,tmax,experiment,
                                                         end_event,min_strength,max_rover_velocity,
                                                         max_cost,max_batt_energy_per_meter)}

Nfeval = 1

def callbackF(Xi):  # this is for SLSQP reporting during optimization
    global Nfeval
    if Nfeval == 1:
        print('Iter        x0         x1        x2        x3         x4           fval')
        
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}  {5: 3.6f} \
          {6: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], obj_f(Xi)))
    Nfeval += 1

print(f'3')



print(f'loop start')

for i in range(1000):
    print(f'loop iter: {i}')

    # initial guess
    x0 = xbest


    # The optimizer options below are
    # 'trust-constr'
    # 'SLSQP'
    # 'differential_evolution'
    # 'COBYLA'
    # You should fully comment out all but the one you wish to use

    ###############################################################################
    #call the trust-constr optimizer --------------------------------------------#

    # options = {'maxiter': 2, 
    #             # 'initial_constr_penalty' : 5.0,
    #             # 'initial_barrier_parameter' : 1.0,
    #             'verbose' : 3,
    #             'disp' : True}
    # res = minimize(obj_f, x0, method='trust-constr', constraints=nonlinear_constraint, 
    #                 options=options, bounds=bounds)
    # end call to the trust-constr optimizer -------------------------------------#
    ###############################################################################

    ###############################################################################
    # call the SLSQP optimizer ---------------------------------------------------#
    options = {'maxiter': 5,
                'disp' : True}
    res = minimize(obj_f, x0, method='SLSQP', constraints=ineq_cons, bounds=bounds, 
                    options=options, callback=callbackF)
    # end call to the SLSQP optimizer --------------------------------------------#
    ###############################################################################

    ###############################################################################
    # call the differential evolution optimizer ----------------------------------#
    # popsize=2 # define the population size
    # maxiter=1 # define the maximum number of iterations
    # res = differential_evolution(obj_f, bounds=bounds, constraints=nonlinear_constraint, popsize=popsize, maxiter=maxiter, disp=True, polish = False) 
    # end call the differential evolution optimizer ------------------------------#
    ###############################################################################

    ###############################################################################
    # call the COBYLA optimizer --------------------------------------------------#
    # cobyla_bounds = [[14, 19], [0.2, 0.7], [250, 800], [0.05, 0.12], [100, 290]]
    # #construct the bounds in the form of constraints
    # cons_cobyla = []
    # for factor in range(len(cobyla_bounds)):
        # lower, upper = cobyla_bounds[factor]
        # l = {'type': 'ineq',
            # 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        # u = {'type': 'ineq',
            # 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        # cons_cobyla.append(l)
        # cons_cobyla.append(u)
        # cons_cobyla.append(ineq_cons)  # the rest of the constraints
    # options = {'maxiter': 50, 
                # 'disp' : True}
    # res = minimize(obj_f, x0, method='COBYLA', constraints=cons_cobyla, options=options)
    # end call to the COBYLA optimizer -------------------------------------------#
    ###############################################################################
    print(f'3.5')


    # check if we have a feasible solution 
    c = constraints_edl_system(res.x,edl_system,planet,mission_events,tmax,experiment,
                            end_event,min_strength,max_rover_velocity,max_cost,
                            max_batt_energy_per_meter)
    feasible = np.max(c - np.zeros(len(c))) <= 0

    # save data

    # Define the header (column names)

    # Design vector elements (in order):
    #   - parachute diameter [m]
    #   - wheel radius [m]
    #   - chassis mass [kg]
    #   - speed reducer gear diameter (d2) [m]
    #   - rocket fuel mass [kg]


    values = {"parachute diameter [m]":res.x[0], 
            "wheel radius [m]":res.x[1], 
            "chassis mass [kg]":res.x[2], 
            "speed reducer gear diameter (d2) [m]":res.x[3], 
            "rocket fuel mass [kg]":res.x[4], 
            "Time":res.fun, 
            "Feasible":feasible,
            "constraint_distance": c[0], 
            "constraint_strength": c[1], 
            "constraint_velocity": c[2], 
            "constraint_cost": c[3], 
            "constraint_battery": c[4]}
    header = list(values.keys())
    
# constraint_distance, constraint_strength, constraint_velocity, constraint_cost, constraint_battery
    # Check if the file exists
    filename = "rover_values2.csv"
    if not os.path.exists(filename):
        # Create the file and write the header
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()  # Write the header
            writer.writerow(values)  # Write the first row of data
        print(f"File {filename} created and data added.")
    else:
        # Open the file in append mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            print(values)
            writer.writerow(values)  # Write the first row of data
        print(f"Data appended to {filename}.")


    if feasible:
        xbest = res.x
        fbest = res.fun
    else:  
        
        print(res.x)
        print(res.fun)
        print(f'constraint_distance, constraint_strength, constraint_velocity, constraint_cost, constraint_battery')
        print(c)
        
        # xbest = [abs(random.triangular(bounds.lb[i], bounds.ub[i], bounds.lb[i] * 1.1)) for i in range(len(bounds.lb))] # reset to random for rerun
        # xbest = [abs(random.uniform(bounds.lb[i], bounds.ub[i])) for i in range(len(bounds.lb))] # reset to random for rerun
        c_pos = [val if (val > 0) else 0 for val in c]
        
        A = np.array([
                    [0.1, 0, 0, 0.1, 0],
                    [0, 0, 0, 0.05, 0],
                    [0, 5, 0, 5, 0],
                    [0, 0, 0, 0.001, 0],
                    [0, 0, 0, 10, 0]
        ])
        
        
        xbest -= A @ c_pos 
        continue
            
        # Design vector elements (in order):
        #   - parachute diameter [m]
        #   - wheel radius [m]
        #   - chassis mass [kg]
        #   - speed reducer gear diameter (d2) [m]
        #   - rocket fuel mass [kg]
        
        #x_lb = np.array([14, 0.2, 250, 0.05, 100])
        #x_ub = np.array([19, 0.7, 800, 0.12, 290])


        fval = [0]
        raise Exception('Solution not feasible, exiting code...')
        sys.exit()
    print(f'4')

    # What about the design variable bounds?

    # The following will rerun your best design and present useful information
    # about the performance of the design
    # This will be helpful if you choose to create a loop around your optimizers and their initializations
    # to try different starting points for the optimization.
    edl_system = redefine_edl_system(edl_system)

    edl_system['parachute']['diameter'] = xbest[0]
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = xbest[1]
    edl_system['rover']['chassis']['mass'] = xbest[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = xbest[3]
    edl_system['rocket']['initial_fuel_mass'] = xbest[4]
    edl_system['rocket']['fuel_mass'] = xbest[4]

    # *****************************************************************************
    # These lines save your design for submission for the rover competition.
    # You will want to change them to match your team information.

    edl_system['team_name'] = 'FunTeamName'  # change this to something fun for your team (or just your team number)
    edl_system['team_number'] = 99    # change this to your assigned team number (also change it below when saving your pickle file)
    print(f'5')

    # This will create a file that you can submit as your competition file.
    with open('FA24_501team99.pickle', 'wb') as handle:
        pickle.dump(edl_system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # *****************************************************************************

    #del edl_system
    #with open('challenge_design_team9999.pickle', 'rb') as handle:
    #    edl_system = pickle.load(handle)
    print(f'6')

    time_edl_run,_,edl_system = simulate_edl(edl_system,planet,mission_events,tmax,True)
    time_edl = time_edl_run[-1]

    edl_system['rover'] = simulate_rover(edl_system['rover'],planet,experiment,end_event)
    time_rover = edl_system['rover']['telemetry']['completion_time']

    total_time = time_edl + time_rover
    
    edl_system_total_cost=get_cost_edl(edl_system)

    print('----------------------------------------')
    print('----------------------------------------')
    print('Optimized parachute diameter   = {:.6f} [m]'.format(xbest[0]))
    print('Optimized rocket fuel mass     = {:.6f} [kg]'.format(xbest[4]))
    print('Time to complete EDL mission   = {:.6f} [s]'.format(time_edl))
    print('Rover velocity at landing      = {:.6f} [m/s]'.format(edl_system['rover_touchdown_speed']))
    print('Optimized wheel radius         = {:.6f} [m]'.format(xbest[1])) 
    print('Optimized d2                   = {:.6f} [m]'.format(xbest[3])) 
    print('Optimized chassis mass         = {:.6f} [kg]'.format(xbest[2]))
    print('Time to complete rover mission = {:.6f} [s]'.format(time_rover))
    print('Time to complete mission       = {:.6f} [s]'.format(total_time))
    print('Average velocity               = {:.6f} [m/s]'.format(edl_system['rover']['telemetry']['average_velocity']))
    print('Distance traveled              = {:.6f} [m]'.format(edl_system['rover']['telemetry']['distance_traveled']))
    print('Battery energy per meter       = {:.6f} [J/m]'.format(edl_system['rover']['telemetry']['energy_per_distance']))
    print('Total cost                     = {:.6f} [$]'.format(edl_system_total_cost))
    print('----------------------------------------')
    print('----------------------------------------')

