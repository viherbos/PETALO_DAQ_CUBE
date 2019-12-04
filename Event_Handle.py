import numpy as np
import os
import pandas as pd
import sys
import time
import json
import pet_graphics as petg


class Event_Handler(object):
    """ Subevent extractor and formatter for NN applications
        Takes MOVIE h5 files and extracts subevents
        CALL returns:
            Gamma1 TIME A array = DATA['DATA_A1']
            Gamma1 TIME B array = DATA['DATA_B1']
            Gamma2 TIME A array = DATA['DATA_A2']
            Gamma2 TIME B array = DATA['DATA_B2']
            Associated Event    = DATA['event']
            Gamma1 first interaction = DATA['TRUE1']
            Gamma2 first interaction = DATA['TRUE2']
    """
    def __init__(self,path,data_filename,json_filename):

        try:
            with open(path+json_filename,'r') as infile:
                self.param = json.load(infile)
        except IOError as e:
            print(e)

        self.SiPM_Matrix = np.reshape(
                            np.arange(self.param['SIPM']['first_sipm'],
                                      self.param['SIPM']['first_sipm']+self.param['SIPM']['n_sipms']),
                                      (self.param['TOPOLOGY']['n_rows'],
                                       self.param['TOPOLOGY']['sipm_ext_row']))

        self.DATA_A           = np.array(pd.read_hdf(path+data_filename,key='DATA_A'))
        self.DATA_B           = np.array(pd.read_hdf(path+data_filename,key='DATA_B'))
        self.EVENTS           = np.array(pd.read_hdf(path+data_filename,key='EVENTS'))
        self.TRUE             = np.array(pd.read_hdf(path+data_filename,key='TRUE'))
        self.sensor_positions = np.array(pd.read_hdf(path+data_filename,key='sensor_positions'))

    def __call__(self, event):
        gamma1_coord = self.TRUE[event,0:3]
        gamma2_coord = self.TRUE[event,3:6]

        #Let's find closest SiPM to gamma1_coord
        aux             = np.ones((self.param['SIPM']['n_sipms'],1))
        gamma1_coord_v  = np.dot(aux,gamma1_coord.reshape(1,3))
        distances       = np.sqrt(np.sum(np.square(self.sensor_positions[:,1:]-gamma1_coord_v),axis=1))
        sipm_g1         = self.sensor_positions[np.argmin(distances),0]

        ring_dim  = self.SiPM_Matrix.shape
        gamma1_matrix_pos = np.where(self.SiPM_Matrix == sipm_g1)
        # Roll SiPM Matrixes to find opposite side of detector
        Xe = np.roll(self.SiPM_Matrix,-gamma1_matrix_pos[1]+ring_dim[1]//4,axis=1)
        # Select opposite side of detector (gamma2)
        Xe_sel = Xe[:,ring_dim[1]//2:]
        Xe_sel_1D = Xe_sel.reshape(-1)
        # Select first side of detector (gamma1)
        Xd_sel = Xe[:,0:ring_dim[1]//2]
        Xd_sel_1D = Xd_sel.reshape(-1)

        # Prepare information to return
        DATA_A_1 = np.zeros(ring_dim)
        DATA_A_2 = np.zeros(ring_dim)
        DATA_B_1 = np.zeros(ring_dim)
        DATA_B_2 = np.zeros(ring_dim)


        for i in range(len(Xd_sel_1D)):
            sipm_coord = np.where(self.SiPM_Matrix == Xd_sel_1D[i])
            DATA_A_1[sipm_coord] = self.DATA_A[event,Xd_sel_1D[i]-self.param['SIPM']['first_sipm']]
            DATA_B_1[sipm_coord] = self.DATA_B[event,Xd_sel_1D[i]-self.param['SIPM']['first_sipm']]

        for i in range(len(Xe_sel_1D)):
            sipm_coord = np.where(self.SiPM_Matrix == Xe_sel_1D[i])
            DATA_A_2[sipm_coord] = self.DATA_A[event,Xe_sel_1D[i]-self.param['SIPM']['first_sipm']]
            DATA_B_2[sipm_coord] = self.DATA_B[event,Xe_sel_1D[i]-self.param['SIPM']['first_sipm']]

        DATA = {}
        DATA['DATA_A1'] = DATA_A_1
        DATA['DATA_A2'] = DATA_A_2
        DATA['DATA_B1'] = DATA_B_1
        DATA['DATA_B2'] = DATA_B_2
        DATA['event']   = self.EVENTS[event]
        DATA['TRUE1']   = gamma1_coord
        DATA['TRUE2']   = gamma2_coord

        return DATA

# Nyapa for pet_graphics compatibility
class SIM_CONT(object):
    def __init__(self,param):
        self.data = param



if __name__ == '__main__':

    #path      = "/mnt/715c6d30-57c4-4aed-a982-551291d8f848/NEUTRINOS/"
    path = "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/"
    data_file = "MOVIE_DATA.h5"
    json_file = "CUBE.json"

    A = Event_Handler(path, data_file, json_file)
    DATA = A(17)


    # GRAPHS
    SIM_CONT = SIM_CONT(A.param)


    graph = petg.data_graph( SIM_CONT,
                                DATA['DATA_B1'].reshape(1,-1),
                                DATA['DATA_B2'].reshape(1,-1),
                                A.sensor_positions,
                                {'photons_id':True,'sipm_id':False})

    graph1 = petg.data_graph( SIM_CONT,
                                DATA['DATA_A1'].reshape(1,-1),
                                DATA['DATA_A2'].reshape(1,-1),
                                A.sensor_positions,
                                {'photons_id':True,'sipm_id':False})
