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
import itertools as it
from SimLib import shared_stuff as ss

# python Simpy_DAQ_CUBE.py -f -d CUBE /home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/


#def L1_exec(SiPM_Matrix_Slice, DATA, timing, TDC, Param):
def L1_exec(SiPM_Matrix_Slice, timing, Param):

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
                             #sim_info    = {'DATA': DATA, 'timing': timing, 'TDC':TDC, 'Param': Param },
                             sim_info    = {'timing': timing, 'Param': Param },
                             SiPM_Matrix_Slice = SiPM_Matrix_Slice)

    DRAIN_instance = DAQ.DATA_drain( out_stream = data_out,
                                            env = env)
    # Wiring
    L1_instance.ETH_OUT.out = DRAIN_instance


    # Run Simulation for a very long time to force flush of FIFOs
    env.run(until = 1E12*Param.P['ENVIRONMENT']['n_events']/Param.P['ENVIRONMENT']['event_rate'])

    OUTPUT_L1      = L1_instance()
    OUTPUT_Drain   = DRAIN_instance()
    OUTPUT_ASICS   = [L1_instance.ASICS[i]() for i in range(n_asics)]
    # Get data for statistics


    return {'DATA_out' : OUTPUT_Drain,
            'L1_out'   : OUTPUT_L1,
            'ASICS_out': OUTPUT_ASICS}


def L1_exec_wrapper(args):
    return L1_exec(*args)


#def DAQ_sim_CUBE( DATA, timing, TDC, Param ):
def DAQ_sim_CUBE( timing, Param ):

    # Generation of Iterable for pool.map
    # Mapping Function
    try:
        style = Param.P['L1']['map_style']
        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(Param.P, style)
    except:
        # JSON file doesn't include mapping option
        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(Param.P, 'striped')

    #L1_exec(L1_Slice[0],sim_info)
    # Multiprocess Pool Management

    #kargs = {'DATA': DATA, 'timing': timing, 'TDC':TDC, 'Param': Param }
    #DAQ_map = partial(L1_exec, **kargs)

    start_time = time.time()
    # Multiprocess Work
    pool_size = mp.cpu_count() #// 2
    pool = mp.Pool(processes=pool_size)

    #pool_output = pool.map(DAQ_map, [i for i in L1_Slice])

    pool_output = pool.map(L1_exec_wrapper, it.izip([i for i in L1_Slice],
                                                    #it.repeat(DATA),
                                                    it.repeat(timing),
                                                    #it.repeat(TDC),
                                                    it.repeat(Param)))

    pool.close()
    pool.join()


    # {'DATA_out': , 'L1_out': , 'ASICS_out':}
    # N Blocks of data (N=n_L1)


    # Create an array with all DATA OUT
    SIM_OUT = []
    lost_producers  = np.array([]).reshape(0,1)
    lost_channels   = np.array([]).reshape(0,1)
    lost_outlink    = np.array([]).reshape(0,1)
    log_channels    = np.array([]).reshape(0,2)
    log_outlink     = np.array([]).reshape(0,2)


    lost_FIFOIN     = np.array([]).reshape(0,1)
    lost_ETHOUT     = np.array([]).reshape(0,1)
    log_FIFOIN      = np.array([]).reshape(0,2)
    log_ETHOUT      = np.array([]).reshape(0,2)
    in_time         = np.array([]).reshape(0,1)
    out_time        = np.array([]).reshape(0,1)

    for L1_i in pool_output:
        for j in range(len(L1_i['DATA_out'])):
            SIM_OUT.append(L1_i['DATA_out'][j])

    # Gather Log information from ASICS layer
    for L1_i in pool_output:
        for j in L1_i['ASICS_out']:
            lost_producers = np.vstack([lost_producers,
                                        np.array(j['lost_producers'])])
            lost_channels = np.vstack([lost_channels,
                                        np.array(j['lost_channels'])])
            lost_outlink  = np.vstack([lost_outlink,
                                        np.array(j['lost_outlink'])])
            log_channels  = np.vstack([log_channels,
                                        np.array(j['log_channels'])])
            log_outlink   = np.vstack([log_outlink,
                                        np.array(j['log_outlink'])])

    # Gather Log information from L1 layer
    for L1_i in pool_output:
        lost_FIFOIN = np.vstack([lost_FIFOIN,
                                    np.array(L1_i['L1_out']['lost_FIFOIN'])])
        lost_ETHOUT = np.vstack([lost_ETHOUT,
                                    np.array(L1_i['L1_out']['lost_ETHOUT'])])
        log_FIFOIN = np.vstack([log_FIFOIN,
                                    np.array(L1_i['L1_out']['log_FIFOIN'])])
        log_ETHOUT = np.vstack([log_ETHOUT,
                                    np.array(L1_i['L1_out']['log_ETHOUT'])])


    pool_output = {'DATA_out' : SIM_OUT,

                   'L1_out'   : {'lost_FIFOIN':lost_FIFOIN,
                                 'lost_ETHOUT':lost_ETHOUT,
                                 'log_FIFOIN':log_FIFOIN,
                                 'log_ETHOUT':log_ETHOUT},

                   'ASICS_out': {'lost_producers':lost_producers,
                                 'lost_channels':lost_channels,
                                 'lost_outlink':lost_outlink,
                                 'log_channels':log_channels,
                                 'log_outlink':log_outlink}
                    }

    elapsed_time = time.time()-start_time
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
    DATA,TDC,sensors,n_events = A.compose()


    # Number of events for simulation
    n_events = CG['ENVIRONMENT']['n_events']

    ss.DATA_g = DATA[0:n_events,:].astype(int)
    ss.TDC_g  = TDC[0:n_events,:].astype(int)

    print (" %d EVENTS IN %d H5 FILES" % (n_events,len(n_files)))



    Param = DAQ.parameters(CG,sensors,n_events)


    # In Christoph we trust
    timing = np.random.poisson(1.0E12/Param.P['ENVIRONMENT']['event_rate'],n_events).astype(int)


    # All sensors are given the same timestamp in an events
    #sim_info = {'DATA': DATA_g, 'timing': timing, 'TDC':TDC_g, 'Param': Param }
    sim_info = {'timing': timing, 'Param': Param }

    # Call Simulation Function
    pool_out,topology = DAQ_sim_CUBE(**sim_info)


    POUT = HF.DAQ_OUT_CUBE(file_name, CG, pool_out, topology)
    POUT.write_raw_out()
    output = POUT.process()
    POUT.write_out(output)








    #//////////////////////////////////////////////////////////////////
    #///                     DATA ANALYSIS AND GRAPHS               ///
    #//////////////////////////////////////////////////////////////////

    graphic_out = HF.CUBE_graphs([file_name],path)
    graphic_out()
