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


class ENERGY_compute(object):

    def __init__(self, path, file_name, SIPM, Matrix_O):
        self.p_name = path
        self.f_name = file_name
        os.chdir(path)
        self.tof_wave = np.array(pd.read_hdf(file_name,key='MC/tof_waveforms'),
                                 dtype='int')
        self.Matrix_O   = Matrix_O
        self.ring_dim   = Matrix_O.shape
        self.n_sipms    = SIPM['n_sipms']
        self.first_sipm = SIPM['first_sipm']
        self.part_acc   = []

    def __call__(self,event):
        try:
            event_select    = np.argwhere(self.tof_wave[:,0]==event)
            event_tof       = self.tof_wave[event_select[:,0],1:]
            event_tof[:,0]  = event_tof[:,0]*-1-self.first_sipm
            # SiPM  |  Timebin  |  Charge
            time_length = np.max(event_tof[:,1])

            pe_table = np.zeros((time_length+1,self.n_sipms))

            for i in range(event_tof.shape[0]):
                pe_table[event_tof[i,1],event_tof[i,0]] = event_tof[i,2]

            part_acc  = np.sum(pe_table,axis=0)
            ev1_sipm  = np.argmax(part_acc)
            ev1_coord = np.where(self.Matrix_O == ev1_sipm)
            # Roll SiPM Matrixes to find opposite side of detector
            Xe = np.roll(self.Matrix_O,-ev1_coord[1]+self.ring_dim[1]//4,axis=1)
            # Select opposite side of detector
            Xe_sel = Xe[:,self.ring_dim[1]//2:]
            Xe_sel_1D = Xe_sel.reshape(-1)
            # Select first side of detector
            Xd_sel = Xe[:,0:self.ring_dim[1]//2]
            Xd_sel_1D = Xd_sel.reshape(-1)

            Xe_ener = np.sum(part_acc[Xe_sel_1D])
            Xd_ener = np.sum(part_acc[Xd_sel_1D])

            # print("E1=%f - E2=%f" % (Xe_ener,Xd_ener))
            # print("EVENT_n: %d" % event)
        except:
            Xe_ener = 0
            Xd_ener = 0
        return np.array([Xe_ener, Xd_ener])



class TOF_compute(object):
    """ SIPM : [risetime_tau (ps), falltime_tau (ps)]
    """
    def __init__(self, path, file_name, SIPM, Matrix_O, time_window, TE_TDC, TE_E, time_bin=5):
        self.p_name = path
        self.f_name = file_name
        # SPE response computation
        self.n_sipms    = SIPM['n_sipms']
        self.first_sipm = SIPM['first_sipm']

        self.Matrix_O = Matrix_O
        self.time_window = time_window
        self.TE_TDC = TE_TDC
        self.TE_E = TE_E
        self.time_bin = time_bin


        tau_sipm   = SIPM['tau_sipm']
        time     = np.arange(0,80000,time_bin)
        alfa = 1.0/tau_sipm[1]
        beta = 1.0/tau_sipm[0]
        t_p = np.log(beta/alfa)/(beta-alfa)
        K = (beta)*np.exp(alfa*t_p)/(beta-alfa)
        self.spe_resp = K*(np.exp(-time*alfa)-np.exp(-time*beta))

        os.chdir(path)

        self.tof_wave   = np.array(pd.read_hdf(file_name,key='MC/tof_waveforms'),
                                   dtype='int')

    def convolve_tof(self,signal,spe_resp):
        conv_first = np.hstack([spe_resp,np.zeros(len(signal)-1)])
        conv_res   = np.zeros(len(signal)+len(spe_resp)-1)
        pe_pos = np.argwhere(signal > 0)
        pe_recov = signal[pe_pos]
        for i in range(len(pe_recov)):
            desp = np.roll(conv_first,pe_pos[i])*pe_recov[i]
            conv_res = desp + conv_res
        return conv_res


    def TDC_first_photon(self, event):


        event_select    = np.argwhere(self.tof_wave[:,0]==event)
        event_tof       = self.tof_wave[event_select[:,0],1:]
        event_tof[:,0]  = event_tof[:,0]*-1-self.first_sipm

        # Beware of empty events
        timestamp_v = np.zeros(self.n_sipms)
        try:

            time_length = np.max(event_tof[:,1])
            #pe_table = np.zeros((time_length+1,self.n_sipms))
            pe_table = np.zeros(self.n_sipms)
            for i in range(event_tof.shape[0]):
                if event_tof[i,1]<timestamp_v[event_tof[i,0]] or timestamp_v[event_tof[i,0]]==0:
                    timestamp_v[event_tof[i,0]] = event_tof[i,1]
                pe_table[event_tof[i,0]] += event_tof[i,2]
            self.part_acc  = pe_table
            # SiPM  |  Timebin  |  Charge


        except:
            self.part_acc  = np.zeros(self.n_sipms)

        return timestamp_v


    def TDC_convolution(self,event):
        event_select    = np.argwhere(self.tof_wave[:,0]==event)
        event_tof       = self.tof_wave[event_select[:,0],1:]
        event_tof[:,0]  = event_tof[:,0]*-1-self.first_sipm
        # SiPM  |  Timebin  |  Charge

        # Beware of Empty Events
        # try:
        #time_length = np.max(event_tof[:,1])
        #print ("Problema: %d" % time_length)
        pe_table = np.zeros((self.time_window,self.n_sipms))

        for i in range(event_tof.shape[0]):
            if event_tof[i,1] < self.time_window:
                pe_table[event_tof[i,1],event_tof[i,0]] = event_tof[i,2]

        #if time_window == -1:
        #    self.conv_table = np.zeros((np.max(event_tof[:,1])+1 + self.spe_resp.shape[0]-1, self.n_sipms))
        #else:
        self.conv_table = np.zeros((pe_table.shape[0] + self.spe_resp.shape[0]-1, self.n_sipms))

        for i in range(self.n_sipms):
            if not np.all(pe_table[:,i]==0): #np.max(pe_table[:,i])>0:
                self.conv_table[:,i] = self.convolve_tof(pe_table[0:self.time_window,i],self.spe_resp)
                #conv_table[:,i] = np.convolve(pe_table[:,i],self.spe_resp)
                #conv_table[:,i] = sc.fftconvolve(pe_table[:,i],self.spe_resp)

        #charge_acc     = np.cumsum(self.conv_table,axis=0)
        #self.part_acc  = np.cumsum(pe_table,axis=0)

        timestamp_v = np.array([])
        for i in range(self.conv_table.shape[1]):
            timestamp  = np.argwhere(self.conv_table[0:self.time_window,i]>self.TE_TDC)
            # We take only the first part up to time_window to speed up the computation
            if timestamp.size == 0:
                timestamp = 0
            else:
                timestamp  = np.min(timestamp)

            timestamp_v = np.hstack([timestamp_v,timestamp])

        self.part_acc = np.zeros(self.n_sipms)
        for i in range(event_tof.shape[0]):
            self.part_acc[event_tof[i,0]] += event_tof[i,2]

        #except:
           # timestamp_v      = np.zeros(self.n_sipms)
           # self.part_acc    = np.zeros(self.n_sipms)
           # self.conv_table  = np.zeros((1,self.n_sipms))

        return timestamp_v


    def __call__(self, event, method, mean):
        t_stamp_v = np.array([]).reshape(0,self.n_sipms)
        ring_dim  = self.Matrix_O.shape
        TOF       = np.array([])
        j=0


        if method=="first_photon":
            timestamp = self.TDC_first_photon(event)
        else:
            timestamp = self.TDC_convolution(event)
        t_stamp_v = np.vstack([t_stamp_v,timestamp])


        #print ("Processing Event : %d" % event)

        timestamp_M    = np.ma.MaskedArray(timestamp,timestamp<1)
        gamma1_sipm    = np.ma.argmin(timestamp_M)
        gamma1_tdc     = np.ma.min(timestamp_M)
        gamma2_sipm    = np.zeros(gamma1_sipm.shape)
        gamma2_tdc     = np.zeros(gamma1_sipm.shape)

        gamma1_coord = np.where(self.Matrix_O==gamma1_sipm)

        # Roll SiPM Matrixes to find opposite side of detector
        Xe = np.roll(self.Matrix_O,-gamma1_coord[1]+ring_dim[1]//4,axis=1)
        # Select opposite side of detector
        Xe_sel = Xe[:,ring_dim[1]//2:]
        Xe_sel_1D = Xe_sel.reshape(-1)

        # Select first side of detector
        Xd_sel = Xe[:,0:ring_dim[1]//2]
        Xd_sel_1D = Xd_sel.reshape(-1)

        Xe_ener = np.sum(self.part_acc[Xe_sel_1D])
        Xd_ener = np.sum(self.part_acc[Xd_sel_1D])

        try:
            OPO_g = timestamp_M[Xe_sel_1D]
            gamma2_tdc = np.ma.min(OPO_g)
            gamma2_coord = Xe_sel_1D[np.ma.argmin(OPO_g)]
        except:
            gamma2_tdc = 0

        # Get rid of singles
        if mean == 0:
            TOF_p = (gamma1_tdc - gamma2_tdc)/2.0
        else:
            gamma1_tdc = np.mean(np.ma.sort(timestamp_M[Xd_sel_1D])[0:mean])
            gamma2_tdc = np.mean(np.ma.sort(timestamp_M[Xe_sel_1D])[0:mean])
            TOF_p = (gamma1_tdc - gamma2_tdc)/2.0

        selec_cond = np.logical_not(np.isnan(TOF_p)) and (Xd_ener>self.TE_E[0]) and (Xd_ener<self.TE_E[1]) \
                     and (Xe_ener>self.TE_E[0]) and (Xe_ener<self.TE_E[1])

        if selec_cond:
            TOF = TOF_p
            print("SiPM1 = %d | SiPM2 = %d | TOF = %f" % (gamma1_sipm, gamma2_coord, TOF_p))
        else:
            TOF = -10000

        return TOF

if __name__ == '__main__':

    # CONFIGURATION READING
    path         = "/volumedisk0/home/viherbos/DAQ_data/"
    jsonfilename = "CUBE"
    SIM_CONT=conf.SIM_DATA(filename=path+jsonfilename+".json",read=True)
    data = SIM_CONT.data
    L1_Slice, Matrix_I, Matrix_O, topo = DAQ.SiPM_Mapping(data,data['L1']['map_style'])

    SIPM = {'n_sipms':3500, 'first_sipm':1000, 'tau_sipm':[100,15000]}
    # Rise time constant limited by external elements (cabling, ASCI impedance)
    # 100 ps is a rough estimation based on crosstalk limit due to feedtrough design
    # Fall time constant is realistic, based on bibliography and measurements

    # GENERAL PARAMETERS
    TE_range = [0.25]
    n_files = 12
    time_bin = 5
    TOF_TE_TDC = []
    mean = 10
    cores = 18

    name = "petit_ring_tof_high_stat"
    path = "/volumedisk0/home/paolafer/vicente/"

    for j in range(len(TE_range)):
        TOF = np.array([])
        start_time = time.time()
        #############################
        for i in range(n_files):
            TDC = TOF_compute(path ,name+"."+str(i).zfill(3)+".pet.h5",
                             SIPM        = SIPM,
                             Matrix_O    = Matrix_O,
                             time_window = 10000,
                             TE_TDC      = TE_range[j],
                             TE_E        = [1000,1600],
                             time_bin    = time_bin)
            event_min = np.min(TDC.tof_wave[:,0])
            event_max = np.max(TDC.tof_wave[:,0])
            # Multiprocessing

            def TOF_comp_wrapper(args):
                return TDC(*args)


            pool_size = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            pool_output = pool.map(TOF_comp_wrapper, zip([i for i in range(event_min,event_max+1)],
                                                     it.repeat("conv"), it.repeat(mean)))
            #time_window, TE_TDC, ev_range, TE_E
            pool.close()
            pool.join()

            # Introduce a random sign to symmetrize distribution
            random_sign = (np.random.rand(len(pool_output))>0.5)*-1
            random_sign = random_sign + random_sign + 1
            TOF = np.hstack([TOF, np.array(pool_output) * random_sign])


        print("---- %s SECONDS FOR TE_TDC = %s------" % (time.time()-start_time, TE_range[j]))
        #################################

        TOF_TE_TDC.append(TOF)



    os.chdir("/volumedisk0/home/viherbos/DAQ_data/")
    with pd.HDFStore("TOF_025_mean.h5",complevel=9, complib='zlib') as store:
        TE_range = pd.DataFrame(data=TE_range)
        TOF_data = pd.DataFrame(data=TOF_TE_TDC)
        store.put('TE_range',TE_range)
        store.put('TOF_data',TOF_data)
        store.close()
