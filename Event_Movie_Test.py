import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from SimLib import config_sim as conf
from SimLib import sipm_mapping as DAQ
import sys
#sys.path.append("/home/viherbos/GITHUB/PETALO_analysis")
import fit_library
import scipy.signal as sc
import itertools as it
import multiprocessing as mp
import time
from antea.reco.reco_functions import find_first_interactions_in_active as antea_ffia
import TOF_test as tof


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



class MOVIE(object):
    """ Create a movie-like register of the gamma1+gamma2 TRUE events
    """
    def __init__(self, path, filename, SiPM, Matrix_O):
        """ path and filename of .h5 data file
            SiPM = [first_sipm, n_sipms]
        """
        self.path = path
        self.filename = filename
        file = path + filename
        particles = pd.read_hdf(file,key='MC/particles')
        hits      = pd.read_hdf(file,key='MC/hits')
        self.keys_particles = particles.columns
        self.keys_hits      = hits.columns
        self.particles      = particles
        self.hits           = hits
        self.tof_wave       = np.array(pd.read_hdf(file,key='MC/tof_waveforms'))
        self.n_sipms        = SiPM['n_sipms']
        self.first_sipm     = SiPM['first_sipm']
        self.Matrix_O     = Matrix_O

    def true_info(self,event):
        """ Gets data from given event
        """
        particles_i = self.particles[self.particles.event_id==event]
        hits_i      = self.hits[self.hits.event_id==event]
        g1,g2,t1,t2 = antea_ffia(particles_i, hits_i)
        data = np.hstack([np.array(g1),np.array(g2),t1,t2])
        return data

    def process_event(self, event, acc_time):
        """ acc_time: Length of Sliding Window Filter (in time_bins)
        """

        with HiddenPrints():
            true_data = np.hstack([event,self.true_info(event)])

        if true_data[1]:
            # Check for valid output
            event_select    = np.argwhere(self.tof_wave[:,0]==event)
            event_tof       = self.tof_wave[event_select[:,0],1:]
            event_tof[:,0]  = event_tof[:,0]*-1-self.first_sipm
            # SiPM  |  Timebin  |  Charge

            # Now Let's find the two sides for gamma1 & gamma2
            TDC = tof.TOF_compute(self.path, self.filename,
                             SIPM        = SIPM,
                             Matrix_O    = self.Matrix_O,
                             time_window = 10000,
                             TE_TDC      = [],
                             TE_E        = [],
                             time_bin    = 5,
                             mean        = [])

            timestamp      = TDC.TDC_first_photon(event)
            timestamp_M    = np.ma.MaskedArray(timestamp,timestamp<1)
            gamma1_sipm    = np.ma.argmin(timestamp_M)
            ring_dim  = self.Matrix_O.shape
            gamma1_coord = np.where(self.Matrix_O==gamma1_sipm)

            # Roll SiPM Matrixes to find opposite side of detector
            Xe = np.roll(self.Matrix_O,-gamma1_coord[1]+ring_dim[1]//4,axis=1)
            # Select opposite side of detector
            Xe_sel = Xe[:,ring_dim[1]//2:]
            Xe_sel_1D = Xe_sel.reshape(-1)

            # Select first side of detector
            Xd_sel = Xe[:,0:ring_dim[1]//2]
            Xd_sel_1D = Xd_sel.reshape(-1)


            time_length = np.max(event_tof[:,1])
            pe_table = np.zeros((time_length+1,self.n_sipms))

            for i in range(event_tof.shape[0]):
                pe_table[event_tof[i,1],event_tof[i,0]] = event_tof[i,2]

            T_photon_event = np.sum(pe_table,axis=1)
            # Sliding Window over Total Photons(time) for this event (computes slope of photon signal)

            slope           = np.convolve(T_photon_event,np.ones(acc_time))
            max_slope_time1 = np.argmax(slope[Xe_sel_1D])
            max_slope_time2 = np.argmax(slope[Xd_sel_1D])
            # time of max arrival of photons/acc_time

            photon_acc  = np.cumsum(pe_table,axis=0)

            # We get the accumulated photons per elapsed time, highest gradient of photons and final photo
            event_info1 = np.zeros(self.n_sipms)
            event_info2 = np.zeros(self.n_sipms)
            event_info1[Xe_sel_1D] = photon_acc[max_slope_time1,Xe_sel_1D]
            event_info2[Xd_sel_1D] = photon_acc[max_slope_time2,Xd_sel_1D]

            event_info = np.hstack([event_info1+event_info2, photon_acc[-1,:]])
            event_info = np.hstack([true_data,event_info])

            print("Event %d is Valid" % event)

        else:
            event_info = np.zeros(9+self.n_sipms*2)-1

        return event_info




if __name__ == '__main__':

    # CONFIGURATION READING
    path         = "/volumedisk0/home/viherbos/DAQ_data/"
    jsonfilename = "CUBE"
    SIM_CONT=conf.SIM_DATA(filename=path+jsonfilename+".json",read=True)
    data = SIM_CONT.data
    L1_Slice, Matrix_I, Matrix_O, topo = DAQ.SiPM_Mapping(data,data['L1']['map_style'])

    SIPM = {'n_sipms':3500, 'first_sipm':1000, 'tau_sipm':[100,15000]}
    # Add SiPM info in .json file

    # GENERAL PARAMETERS
    n_files = 50
    time_bin = 5
    cores = 20

    name = "petit_ring_tof_all_tables"
    path = "/volumedisk0/home/paolafer/vicente/"


    DATA = np.array([]).reshape(0, 9 + SIPM['n_sipms']*2)
    start_time = time.time()
    #############################
    for i in range(n_files):
        M = MOVIE(path ,name+"."+str(i).zfill(3)+".pet.h5", SiPM = SIPM, Matrix_O=Matrix_O)
        event_min = np.min(M.tof_wave[:,0])
        event_max = np.max(M.tof_wave[:,0])
        # Multiprocessing

        def movie_wrapper(args):
            return M.process_event(*args)

        #movie_wrapper([1,200])

        pool_size = mp.cpu_count()
        pool = mp.Pool(processes=cores)
        pool_output = pool.map(movie_wrapper, zip([i for i in range(event_min,event_max+1)],
                                                 it.repeat(200)))
        pool.close()
        pool.join()


        pool_output = np.array(pool_output)
         #.reshape(len(pool_output),8 + SIPM['n_sipms'])

        print("---- %s SECONDS FOR FILE = %s------" % (time.time()-start_time, i))
        #################################


        DATA=np.vstack([DATA,pool_output[pool_output[:,0]>-1, :]])


    sensor_positions = np.array(pd.read_hdf(path+name+".000.pet.h5",key='MC/sensor_positions'))

    os.chdir("/volumedisk0/home/viherbos/DAQ_data/")
    with pd.HDFStore("MOVIE_DATA.h5",complevel=9, complib='zlib') as store:

        DATA_A = pd.DataFrame(data = DATA[:,9:SIPM['n_sipms']+9])
        DATA_B = pd.DataFrame(data = DATA[:,SIPM['n_sipms']+9:])
        TRUE   = pd.DataFrame(data = DATA[:,1:9])
        EVENTS = pd.DataFrame(DATA[:,0])

        sensor_positions = pd.DataFrame(data = sensor_positions)

        store.put('EVENTS',EVENTS)
        store.put('TRUE',TRUE)
        store.put('DATA_A',DATA_A)
        store.put('DATA_B',DATA_B)
        store.put('sensor_positions',sensor_positions)

        store.close()
