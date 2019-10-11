import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_analysis/")
import fit_library
#import HF_translator as HFT
import os
import multiprocessing as mp
from functools import partial
from SimLib import DAQ_infinity as DAQ
from SimLib import HF_files as HF
from SimLib import sipm_mapping as MAP
import time
from SimLib import config_sim as CFG
#from SimLib import pet_graphics as PG
import pandas as pd
import math
import argparse
import tables as tb


# python Simpy_DAQ_CUBE.py -f -d CUBE /home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/


def L1_exec(SiPM_Matrix_Slice, sim_info):
    """ Executes L1 behavior in Simulation
        Input:  SiPM_Matrix_Slice
        Output: { 'DATA_out' : Output data stream,
                  'L1_out'   : {'lost_FIFOIN':,
                                'lost_ETHOUT':,
                                'log_FIFOIN':,
                                'log_ETHOUT':},
                  'ASICS_out': {'lost_producers':,
                                'lost_channels':,
                                'lost_outlink':,
                                'log_channels':
                                'log_outlink':}
                }
    """
    env      = simpy.Environment()
    n_asics  = len(SiPM_Matrix_Slice)
    data_out = []

    # Create instance of L1
    L1_instance    = DAQ.L1( env         = env,
                             sim_info    = sim_info,
                             SiPM_Matrix_Slice = SiPM_Matrix_Slice)

    DRAIN_instance = DAQ.DATA_drain( out_stream = data_out,
                                            env = env)
    # Wiring
    L1_instance.ETH_OUT.out = DRAIN_instance


    # Run Simulation for a very long time (100sec) to force flush of FIFOs
    env.run(until = 100E9)

    OUTPUT_L1      = L1_instance()
    OUTPUT_Drain   = DRAIN_instance()
    OUTPUT_ASICS   = [L1_instance.ASICS[i]() for i in range(n_asics)]
    # Get data for statistics


    return {'DATA_out' : OUTPUT_Drain,
            'L1_out'   : OUTPUT_L1,
            'ASICS_out': OUTPUT_ASICS}


def DAQ_sim_CUBE(sim_info):
    param = sim_info['Param']

    # Generation of Iterable for pool.map
    # Mapping Function
    try:
        style = param.P['L1']['map_style']
        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(param.P, style)
    except:
        # JSON file doesn't include mapping option
        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(param.P, 'striped')

    L1_exec(L1_Slice[0],sim_info)
    # Multiprocess Pool Management
    kargs = {'sim_info':sim_info}
    DAQ_map = partial(L1_exec, **kargs)

    start_time = time.time()
    # Multiprocess Work
    pool_size = mp.cpu_count() #// 2
    pool = mp.Pool(processes=pool_size)

    pool_output = pool.map(DAQ_map, [i for i in L1_Slice])

    pool.close()
    pool.join()
    elapsed_time = time.time()-start_time

    # {'DATA_out': , 'L1_out': , 'ASICS_out':}
    # N Blocks of data (N=n_L1)

    print ("SKYNET GAINED SELF-AWARENESS AFTER %d SECONDS" % elapsed_time)


    return pool_output,topology




if __name__ == '__main__':

    # Argument parser for config file name
    parser = argparse.ArgumentParser(description='PETALO Infinity DAQ Simulator.')
    parser.add_argument("-f", "--json_file", action="store_true",
                        help="Simulate with configuration stored in json file")
    parser.add_argument('arg1', metavar='N', nargs='?', help='')
    parser.add_argument("-d", "--directory", action="store_true",
                        help="Work directory")
    parser.add_argument('arg2', metavar='N', nargs='?', help='')
    args = parser.parse_args()

    if args.json_file:
        file_name = ''.join(args.arg1)
    else:
        file_name = "sim_config"
    if args.directory:
        path = ''.join(args.arg2)
    else:
        path="./"

    config_file = file_name + ".json"

    CG = CFG.SIM_DATA(filename = path + config_file, read = True)
    CG = CG.data
    # Read data from json file

    n_sipms_int = CG['TOPOLOGY']['sipm_int_row']*CG['TOPOLOGY']['n_rows']
    n_sipms_ext = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_int + n_sipms_ext
    first_sipm  = CG['TOPOLOGY']['first_sipm']

    n_files = CG['ENVIRONMENT']['n_files']
    # Number of files to group for data input
    A = HF.hdf_compose( CG['ENVIRONMENT']['path_to_files'],
                        CG['ENVIRONMENT']['file_name'],
                        n_files,n_sipms)
    DATA,sensors,n_events = A.compose()


    # Number of events for simulation
    n_events = CG['ENVIRONMENT']['n_events']
    DATA = DATA[0:n_events,:]
    print (" %d EVENTS IN %d H5 FILES" % (n_events,len(n_files)))



    Param = DAQ.parameters(CG,sensors,n_events)


    # In Christoph we trust
    timing = np.random.poisson(1E9/Param.P['ENVIRONMENT']['event_rate'],n_events).astype(int)


    # All sensors are given the same timestamp in an events
    sim_info = {'DATA': DATA, 'timing': timing, 'Param': Param }

    # Call Simulation Function
    pool_out,topology = DAQ_sim_CUBE(sim_info)


    POUT = HF.DAQ_OUT_CUBE(file_name, CG, pool_out, topology)
    POUT.write_raw_out()
    output = POUT.process()
    POUT.write_out(output)








    #//////////////////////////////////////////////////////////////////
    #///                     DATA ANALYSIS AND GRAPHS               ///
    #//////////////////////////////////////////////////////////////////

    graphic_out = HF.CUBE_graphs([file_name],path)
    graphic_out()
