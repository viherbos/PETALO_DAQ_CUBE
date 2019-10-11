import json
import os
import numpy as np
import sys
import pandas as pd




class SIM_DATA(object):

    # Only filenames are read. The rest is taken from json file
    def __init__(self,filename="sim_config.json",read=True):
        self.filename = filename
        self.data=[]

        if (read==True):
            self.config_read()
        else:
            # These are default values.
            # L1 output data frame = QDC[10] + TDC[10] + SiPM[20] = 40 bits
            self.data= {'ENVIRONMENT'  :{'event_rate'     :0.7E6,
                                        #'event_rate'     :7.1E6,
                                        'temperature' :300,
                                        'path_to_files': "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/",
                                        'file_name': "p_RING_7mm_6x6_",
                                        'MC_file_name':"full_ring_iradius165mm_z140mm_depth3cm_pitch7mm",
                                        'out_file_name':"DAQ_OF_7mm",
                                        'MC_out_file_name':"FASTDAQ_OF_7mm",
                                        'time_bin': 5,
                                        'n_files' : range(1),
                                        'n_events': 5000},

                        'SIPM'        :{'size'        :[1,6,6]},

                        'TOPOLOGY'    :{'radius_int'   :0,
                                        'radius_ext'   :165,
                                        'sipm_int_row' :0,
                                        'sipm_ext_row' :175,
                                        'n_rows'       :20,
                                        'first_sipm'   :1000},

                        'TOFPET'      :{'n_channels'  :64,
                                        'outlink_rate': (2.6E9/80)/2.0,
                                        # 80 bits per TOFPET output frame
                                        'IN_FIFO_depth':4,
                                        'OUT_FIFO_depth':64*4,
                                        'MAX_WILKINSON_LATENCY':5120,
                                        'TE':2,
                                        'TGAIN':1},

                        'L1'          :{'n_asics'       :10,
                                        'L1_outrate'    :1000E6,
                                        'FIFO_L1a_depth':32,
                                        'FIFO_L1a_freq' :400E6,
                                        'FIFO_L1b_depth':512,
                                        'FIFO_L1b_freq' :400E6,
                                        'n_asics'       :10,
                                        'TE'            :3,
                                        'map_style'     :'striped_3',
                                        'L1_mapping_I'  :[],
                                        'L1_mapping_O'  :[9,9,9,10,9,9],
                                        }
                       }
# 'L1_mapping_O'  :[11,12,12,12,12,12]
# 'L1_mapping_I'  :[10,10,10,10],
# 'L1_mapping_O'  :[10,10,11,10,10]}
# 'L1_mapping_I'  :[5,5,5,5,5,5,5,5],
# 'L1_mapping_O'  :[6,7,6,7,6,6,7,6]}

# 'L1_mapping_I'  :[8,8,8,8,8],
# 'L1_mapping_O'  :[7,7,8,7,7,7,8]}

    def config_write(self):
        writeName = self.filename
        try:
            with open(writeName,'w') as outfile:
                json.dump(self.data, outfile, indent=4, sort_keys=False)
                #print self.data
        except IOError as e:
            print(e)

    def config_read(self):
        try:
            with open(self.filename,'r') as infile:
                self.data = json.load(infile)
                #print self.data
        except IOError as e:
            print(e)



if __name__ == '__main__':

    #filename = "OF_4mm_BUF640_V3"
    filename = "CUBE"
    SIM=SIM_DATA(filename = "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/"+filename+".json",
                 read = False)
    SIM.config_write()
