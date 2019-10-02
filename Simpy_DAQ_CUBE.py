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
from SimLib import Encoder_tools as ET


# def L1_sch_WP(SiPM_Matrix_Slice, sim_info):
#
#     data_out   = []
#     param = sim_info['Param']
#     DATA  = sim_info['DATA']
#
#     env = simpy.Environment()
#
#     n_asics = len(SiPM_Matrix_Slice)
#
#     L1 = DAQ.L1_WAVELET( env = env,
#                  out_stream  = data_out,
#                  param       = param,
#                  SiPM_Matrix = SiPM_Matrix_Slice)
#
#     block_size = param.P['TOFPET']['n_channels']
#
#     ASICS_L1 = [DAQ.FE_asic(
#                     env     = env,
#                     param   = param,
#                     data    = DATA[:,SiPM_Matrix_Slice[i]],
#                     timing  = sim_info['timing'],
#                     sensors = sim_info['Param'].sensors[SiPM_Matrix_Slice[i]],
#                     asic_id = i)
#                 for i in range(n_asics)]
#
#     for i in range(len(ASICS_L1)):
#         ASICS_L1[i].Link.out = L1
#
#     # Run Simulation for a very long time to force flush of FIFOs
#     env.run(until = 100E9)
#
#     OUTPUT_L1     = L1()
#     OUTPUT_ASICS  = [ASICS_L1[i]() for i in range(n_asics)]
#
#     print "L1 finished"
#
#     return {'L1_out':OUTPUT_L1, 'ASICS_out':OUTPUT_ASICS}


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
                             SiPM_Matrix = SiPM_Matrix_Slice)

    DRAIN_instance = DAQ.DATA_drain(env        = env,
                                    out_stream = data_out)

    # Run Simulation for a very long time (100sec) to force flush of FIFOs
    env.run(until = 100E9)

    OUTPUT_L1      = L1_instance()
    OUTPUT_Drain   = DRAIN_instance()
    OUTPUT_ASICS   = [L1_instance.ASICS[i]() for i in range(n_asics)]
    # Get data for statistics

    print "L1 finished"

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

    # Multiprocess Pool Management
    kargs = {'sim_info':sim_info}
    DAQ_map = partial(L1_sch_WP, **kargs)

    start_time = time.time()
    # Multiprocess Work
    pool_size = mp.cpu_count() #// 2
    pool = mp.Pool(processes=pool_size)

    pool_output = pool.map(DAQ_map, [i for i in L1_Slice])

    pool.close()
    pool.join()
    elapsed_time = time.time()-start_time
    #pool_output = DAQ_map(L1_Slice[0])

    print ("SKYNET GAINED SELF-AWARENESS AFTER %d SECONDS" % elapsed_time)


    return pool_output,topology



def DAQ_OUTPUT_processing_CUBE(SIM_OUT,n_L1,n_asics,CG):
    data, in_time, out_time, L1_id, lostL1a, lostL1b = [],[],[],[],[],[]
    lost_producers= np.array([]).reshape(0,1)
    lost_channels = np.array([]).reshape(0,1)
    lost_outlink = np.array([]).reshape(0,1)
    SIM_OUT_L1      = np.array(SIM_OUT['L1_out'])
    SIM_OUT_ASICs   = np.array(SIM_OUT['ASICS_out'])
    logL1_IN = np.array([]).reshape(0,2)
    logL1_OUT = np.array([]).reshape(0,2)
    #logC = np.array([]).reshape(0,2)
    log_channels = np.array([]).reshape(0,2)
    log_outlink = np.array([]).reshape(0,2)
    #L1_frag     = np.array([]).reshape(0,n_L1)
    #L1_frag_aux = np.zeros((1,n_L1),dtype=int)


    # Gather information from ASICS layer
    for j in range(n_asics):
        lost_producers = np.vstack([lost_producers,
                                    SIM_OUT_ASICs[j]['lost_producers']])
        lost_channels = np.vstack([lost_channels,
                                    SIM_OUT_ASICs[j]['lost_channels']])
        lost_outlink  = np.vstack([lost_outlink,
                                    SIM_OUT_ASICs[j]['lost_outlink']])
        log_channels  = np.vstack([log_channels,
                                    SIM_OUT_ASICs[j]['log_channels']])
        log_outlink   = np.vstack([log_outlink,
                                    SIM_OUT_ASICs[j]['log_outlink']])

    # Gather information from L1 layer
    for j in range(n_L1):

        #lostL1a.append(SIM_OUT_L1[j]['lostL1a'])
        lostL1b.append(SIM_OUT_L1[j]['logETH_OUT'])
        logL1_IN  = np.vstack([logL1_IN,SIM_OUT_L1[j]['logFIFO_IN']])
        logL1_OUT = np.vstack([logL1_OUT,SIM_OUT_L1[j]['logETH_OUT']])
        #logC=np.vstack([logC,SIM_OUT_L1[j]['logC']])

        for i in range(len(SIM_OUT_L1[j]['data_out'])):
            #if SIM_OUT[j]['data_out'][i]['data'][0] > 0:
            data.append(SIM_OUT_L1[j]['data_out'][i]['data'])
            in_time.append(SIM_OUT_L1[j]['data_out'][i]['in_time'])
            out_time.append(SIM_OUT_L1[j]['data_out'][i]['out_time'])
            L1_id.append(j)

    #
    # A = np.array(data)
    # sort = np.array([i[1] for i in A])
    # sort_res = np.argsort(sort)
    # A = A[sort_res]
    # L1_id = np.array(L1_id)
    # L1_id = L1_id[sort_res]
    #
    # in_time = np.array(in_time)
    # in_time = in_time[sort_res]
    # out_time = np.array(out_time)
    # out_time = out_time[sort_res]
    #
    # n_TDC = np.array([])
    # i_TDC = np.array([])
    # TDC = np.array([A[i][1] for i in range(len(A))])
    #
    #
    # prev=0
    # for i in TDC:
    #     if (i != prev):
    #         cond = np.array((TDC == i))
    #         n_TDC = np.concatenate((n_TDC,[np.sum(cond)]),axis=0)
    #         i_TDC = np.concatenate((i_TDC,[i]),axis=0)
    #         # For each TDC==i take all L1_id
    #
    #         selec = np.array(range(len(cond)))
    #
    #         for j in selec[cond]:
    #             #print ("%d - %d" % (i,L1_id[j]))
    #             L1_frag_aux[0,L1_id[j]] += 1
    #
    #         #print L1_frag_aux
    #
    #         if (np.sum(L1_frag_aux[0,:]) != np.sum(cond)):
    #             print ("FRAGMENTATION ASSERTION")
    #         L1_frag = np.vstack([L1_frag,L1_frag_aux])
    #
    #         L1_frag_aux = np.zeros((1,n_L1),dtype=int)
    #         prev = i
    # # Scan TDC list : n_TDC number of dataframes with same i_TDC
    # # i_TDC list of different TDC
    #
    #
    #
    # # Timestamp Generator
    # n_bits = np.zeros(len(A))
    # # Buffer compression statistic
    # for i in range(len(A)):
    #     n_bits[i] = DAQ.L1_outframe_nbits_WAV(A[i],CG)
    #
    # time_vector = np.add.accumulate(timing)
    # #Event number location
    # event_order = []
    # cnt = 0
    # for i in i_TDC:
    #     locked = np.argwhere(time_vector==i)
    #     for j in locked:
    #         # Sometimes we have the same TDC for consecutive events
    #         event_order.append(time_vector[int(j)])
    #         cnt += 1
    # event_order = np.array(event_order)
    #
    #
    # # Data table building
    # event = 0
    # A_index = 0
    #
    # WP_len = CG['L1']['wav_blocksize']
    # bs     = CG['L1']['wav_blocksize']
    # n_WP = WP_len * n_L1
    #
    # data_LL = np.zeros((n_events,n_WP),dtype='float')
    # data_LH = np.zeros((n_events,n_WP),dtype='float')
    # data_HL = np.zeros((n_events,n_WP),dtype='float')
    #
    # for i in i_TDC:
    #     for j in range(int(n_TDC[event])):
    #         ind_t = event_order[event]
    #         #data[np.argwhere(time_vector==ind_t),int(A[A_index][2*l+2])] = A[A_index][2*l+3]
    #         L1_id2 = L1_id[A_index]
    #         data_LL[np.argwhere(time_vector==ind_t),int(L1_id2*WP_len):int((L1_id2+1)*WP_len)] = A[A_index][2:bs+2]
    #         data_LH[np.argwhere(time_vector==ind_t),int(L1_id2*WP_len):int((L1_id2+1)*WP_len)] = A[A_index][bs+2:2*bs+2]
    #         data_HL[np.argwhere(time_vector==ind_t),int(L1_id2*WP_len):int((L1_id2+1)*WP_len)] = A[A_index][2*bs+2:3*bs+2]
    #
    #         A_index += 1
    #
    #     event += 1





    output = {'data':[data_LL,data_LH,data_HL],

              'L1': {'in_time': in_time, 'out_time': out_time,
                     'lostL1b': lostL1b, 'logA': logA, 'logB': logB,
                     'logC': logC, 'frag':np.hstack([np.transpose([i_TDC]),L1_frag])},

              'ASICS':{ 'lost_producers':lost_producers,
                        'lost_channels':lost_channels,
                        'lost_outlink':lost_outlink,
                        'log_channels':log_channels,
                        'log_outlink':log_outlink},

              'compress': n_bits,

              'tstamp_event':event_order,

              'timestamp':time_vector
            }


    return output



# def q_d(data,bits,FS):
#     bins = np.arange(FS[0],FS[1],float(FS[1]-FS[0])/(2**bits),dtype=float)
#     data[data>bins[-1]] = FS[1]-float(FS[1]-FS[0])/(2**bits)
#     data[data<bins[0]] = FS[0]
#     return bins[np.digitize(data,bins,right=True)]
#


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


    # # Keras Model read hack:
    # encoder_file = CG['ENVIRONMENT']['AUTOENCODER_file_name']
    # with tb.open_file(path + encoder_file) as h5file:
    #     B=[]
    #     for array in h5file.walk_nodes("/"):
    #         B.append(array)
    #     COMP={}
    #     COMP={'ENC_bias_A'    :np.array([B[-2][:]],dtype=float).T,
    #           'ENC_weights_A' :np.array(B[-1][:],dtype=float),
    #           'DEC_bias_A'    :np.array([B[-4][:]],dtype=float).T,
    #           'DEC_weights_A' :np.array(B[-3][:],dtype=float)}



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

    # SHOW = PG.DET_SHOW(CG.data)
    # os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    # filename = "p_FRSET_0.h5"
    # positions = np.array(pd.read_hdf(filename,key='sensors'))
    # data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    # SHOW(positions,data,0,True,False)


    Param = DAQ.parameters(CG,sensors,n_events)


    # In Christoph we trust
    timing = np.random.poisson(1E9/Param.P['ENVIRONMENT']['ch_rate'],n_events).astype(int)


    # All sensors are given the same timestamp in an events
    sim_info = {'DATA': DATA, 'timing': timing, 'Param': Param }

    # Call Simulation Function
    pool_out,topology = DAQ_sim_WAVELET(sim_info)

    # Translate Simulation Output into an array for Data recovery
    SIM_OUT = {'L1_out':[], 'ASICS_out':[]}
    for i in range(len(pool_out)):
        SIM_OUT['L1_out'].append(pool_out[i]['L1_out'])
        for j in range(len(pool_out[i]['ASICS_out'])):
            SIM_OUT['ASICS_out'].append(pool_out[i]['ASICS_out'][j])

    # Data Output recovery
    n_L1    = np.array(CG['L1']['L1_mapping_O']).shape[0]
    n_asics = np.sum(np.array(CG['L1']['L1_mapping_O']))
    #topology['n_L1'],topology['n_asics'])


    out = DAQ_OUTPUT_processing_WAVELET(SIM_OUT,n_L1,n_asics,CG)



    ####################  WAVELET PACKET PROCESSING ##########################

    n_rows = CG['TOPOLOGY']['n_rows']
    kwargs = {'n_rows':n_rows,
              'TE2':CG['L1']['TE'],
              'n_sensors':0,
              'base':CG['L1']['wav_base']}

    # Quantize and clip
    T_LL = CG['L1']['TW'][0]
    T_LH = CG['L1']['TW'][1]
    T_HL = CG['L1']['TW'][2]
    Q_LL = CG['L1']['QW'][0]
    Q_LH = CG['L1']['QW'][1]
    Q_HL = CG['L1']['QW'][2]

    WAV = ET.encoder_tools_PW(**kwargs)
    WAV.L1_size_compressed =[8,20]

    L1_box, SiPM_I, SiPM_O, topology = MAP.SiPM_Mapping(CG,CG['L1']['map_style'])
    data_recons    = np.zeros((n_events,n_sipms),dtype='float')


    # data_encoded = np.array(pd.read_hdf("/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/5mm_pitch/VER5/FASTDAQOUT_OF5mm_TENC150.000.h5",
    #                         key='MC_encoded'), dtype = 'float')

    for i in range(n_events):
        index_1 = 0
        for L1 in L1_box:
            # Build L1_SiPM matrix
            L1_SiPM = np.array([],dtype='int').reshape(n_rows,0)
            for asic in L1:
                L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((n_rows,-1),order='F')))

            data_LL_aux = out['data'][0][i,:]
            data_LH_aux = out['data'][1][i,:]
            data_HL_aux = out['data'][2][i,:]
            #data_enc_aux = data_encoded[i,:]

            # Quantize and clip + Threshold
            data_LL_aux = q_d(data_LL_aux,Q_LL,[0,1200])
            data_LH_aux = q_d(data_LH_aux,Q_LH,[-600,600])
            data_HL_aux = q_d(data_HL_aux,Q_HL,[-600,600])
            data_LL_aux[np.abs(data_LL_aux)<T_LL] = 0
            data_LH_aux[np.abs(data_LH_aux)<T_LH] = 0
            data_HL_aux[np.abs(data_HL_aux)<T_HL] = 0


            index_1,recons_event = WAV.decoder( L1_SiPM,
                                                data_LL_aux,
                                                data_LH_aux,
                                                data_HL_aux,
                                                index_1)

            for sipm_id in L1_SiPM:
                # data_recons_event is now a matrix with same shape as L1_SiPM (see below)
                data_recons[i,sipm_id] = recons_event[L1_SiPM==sipm_id]

    #data_recons = data_recons * (data_recons > CG['L1']['TE'])


    #####################################################################

    # Write output to file
    cfg_filename = file_name[file_name.rfind("/")+1:]
    DAQ_dump = HF.DAQ_IO(CG['ENVIRONMENT']['path_to_files'],
                    CG['ENVIRONMENT']['file_name'],
                    CG['ENVIRONMENT']['file_name']+"000.h5",
                    CG['ENVIRONMENT']['out_file_name']+"_"+cfg_filename+".h5")
    logs = {  'logA':out['L1']['logA'],
              'logB':out['L1']['logB'],
              'logC':out['L1']['logC'],
              'frame_frag':out['L1']['frag'],
              'log_channels':out['ASICS']['log_channels'],
              'log_outlink': out['ASICS']['log_outlink'],
              'in_time': out['L1']['in_time'],
              'out_time': out['L1']['out_time'],
              'lost':{  'producers':out['ASICS']['lost_producers'].sum(),
                        'channels' :out['ASICS']['lost_channels'].sum(),
                        'outlink'  :out['ASICS']['lost_outlink'].sum(),
                        'L1b'      :np.array(out['L1']['lostL1b']).sum()
                      },
              'compress':out['compress'],
              'tstamp_event':out['tstamp_event'],
              'timestamp':out['timestamp']
            }
    DAQ_dump.write_out(data_recons,topology,logs)




    #//////////////////////////////////////////////////////////////////
    #///                     DATA ANALYSIS AND GRAPHS               ///
    #//////////////////////////////////////////////////////////////////

    graphic_out = HF.infinity_graphs_WP([file_name],path)
    graphic_out()
