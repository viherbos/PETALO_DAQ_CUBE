import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_DAQ_infinity/SimLib")
sys.path.append("../PETALO_analysis")
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import fit_library
import config_sim as CFG
import DAQ_infinity as DAQ
from matplotlib.ticker import MaxNLocator
import scipy.io as SCIO
import sipm_mapping as SM
import string
from SimLib import sipm_mapping as MAP


class DAQ_OUT_CUBE(object):
    """
    DAQ output writing for generic data
    """
    def __init__(self, cfg_filename, CG, pool_out, topology):
        self.path            = CG['ENVIRONMENT']['path_to_files']
        self.daq_outfile     = CG['ENVIRONMENT']['out_file_name']+"_"+cfg_filename+".h5"
        self.daq_outfile_raw = CG['ENVIRONMENT']['out_file_name']+"_"+cfg_filename+"_RAW.h5"

        os.chdir(self.path)
        self.sensors_xyz = np.array( pd.read_hdf(CG['ENVIRONMENT']['file_name']+"000.h5",
                                    key='sensors'),
                                    dtype = 'float32')
        self.topology = topology
        self.pool_out = pool_out

        self.n_L1       = np.array(CG['L1']['L1_mapping_O']).shape[0]
        self.n_asics    = np.sum(np.array(CG['L1']['L1_mapping_O']))
        self.n_sipm     = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
        self.first_sipm = CG['TOPOLOGY']['first_sipm']
        self.n_events    = CG['ENVIRONMENT']['n_events']


    def write_raw_out(self):
        # RAW data will be written in out_timt order, as generated in the real DAQ

        # data_sot: list of ch_frame dictionaries sorted by out time (sot)
        data_sot = sorted(self.pool_out['DATA_out'], key=lambda k:k['out_time'])

        keys = data_sot[0].keys()
        data = np.array(list(data_sot[j].values()
                        for j in range(len(data_sot)) ))

        topo_data = np.array(list(self.topology.values())).reshape(1,len(list(self.topology.values())))

        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile_raw,
                        complevel=9, complib='bzip2') as store:

            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            raw_data      = pd.DataFrame(data = data, columns=keys)
            topo          = pd.DataFrame(data = topo_data,
                                         columns = list(self.topology.keys()))

            store.put('raw',raw_data)
            store.put('sensors',sensors_array)
            store.put('topology',topo)
            store.close()

    def process(self,sensor_0=1000):

        """ POOL_OUT = Simulation Pool Output
            {'DATA_out' : ch_frame array,
             'L1_out'   : L1 logs,
             'ASICS_out': ASICS logs}
        """

        # data_sit: list of ch_frame dictionaries sorted by in time (sit)
        data_sit = sorted(self.pool_out['DATA_out'], key=lambda k:k['in_time'])


        data_panel = np.zeros([self.n_events,self.n_sipm],dtype=int)

        in_time         = np.zeros([self.n_events,self.n_sipm],dtype=int)
        out_time        = np.zeros([self.n_events,self.n_sipm],dtype=int)


        # Fill the table
        # index = 0
        # count = 0

        for i in range(np.shape(data_sit)[0]):
            sipm  = data_sit[i]['sensor_id']-sensor_0
            event = data_sit[i]['event']
            in_time[event,sipm]  = data_sit[i]['in_time']
            out_time[event,sipm] = data_sit[i]['out_time']
            data_panel[event,sipm] =+ data_sit[i]['data']
        # for i in range(self.n_events):
        #     timestamp = data_sit[index]['in_time']
        #     in_time   = np.vstack([in_time,timestamp])
        #     # For a given event the time of input to the system is the timestamp
        #     # but the output time is the time of the last output related to that event
        #
        #     out_stamp = 0
        #     while timestamp == data_sit[count]['in_time']:
        #         sensor_id = data_sit[count]['sensor_id']-1000
        #         data_panel[i][sensor_id] = data_sit[count]['data']
        #
        #         if out_stamp < data_sit[count]['out_time']:
        #             out_stamp = data_sit[count]['out_time']
        #         # Take the worst output timestamp
        #
        #         if count < len(data_sit)-1:
        #             count = count + 1
        #         else:
        #             break
        #
        #     out_time  = np.vstack([out_time, out_stamp])
        #
        #     index = count



        output = {'QDC_table': data_panel,
                  'L1_log'   : self.pool_out['L1_out'],
                  'ASICS_log': self.pool_out['ASICS_out'],
                  'in_time' : in_time,
                  'out_time': out_time
                }


        return output


    def write_out(self, p_out):

        topo_data = np.array(list(self.topology.values())).reshape(1,len(list(self.topology.values())))

        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile,
                        complevel=9, complib='bzip2') as store:

            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            qdc_data      = pd.DataFrame(data = np.array(p_out['QDC_table']),
                                         columns=self.sensors_xyz[:,0])
            topo          = pd.DataFrame(data = topo_data,
                                         columns = list(self.topology.keys()))

            lost_producers = pd.DataFrame(data = p_out['ASICS_log']['lost_producers'])
            lost_channels  = pd.DataFrame(data = p_out['ASICS_log']['lost_channels'])
            lost_asicout   = pd.DataFrame(data = p_out['ASICS_log']['lost_outlink'])
            log_channels   = pd.DataFrame(data = p_out['ASICS_log']['log_channels'])
            log_asicout    = pd.DataFrame(data = p_out['ASICS_log']['log_outlink'])

            lost_FIFOIN   = pd.DataFrame(data = p_out['L1_log']['lost_FIFOIN'])
            lost_ETHOUT   = pd.DataFrame(data = p_out['L1_log']['lost_ETHOUT'])
            log_FIFOIN    = pd.DataFrame(data = p_out['L1_log']['log_FIFOIN'])
            log_ETHOUT    = pd.DataFrame(data = p_out['L1_log']['log_ETHOUT'])

            in_time       = pd.DataFrame(data = p_out['in_time'])
            out_time      = pd.DataFrame(data = p_out['out_time'])

            store.put('qdc_data',qdc_data)
            store.put('sensors',sensors_array)
            store.put('topology',topo)
            store.put('in_time',in_time)
            store.put('out_time',out_time)

            store.put('lost_producers',lost_producers)
            store.put('lost_channels',lost_channels)
            store.put('lost_asicout',lost_asicout)
            store.put('lost_FIFOIN',lost_FIFOIN)
            store.put('lost_ETHOUT',lost_ETHOUT)

            store.put('log_channels',log_channels)
            store.put('log_asicout',log_asicout)
            store.put('log_FIFOIN',log_FIFOIN)
            store.put('log_ETHOUT',log_ETHOUT)
            store.close()



class hdf_access(object):
    """ A utility class to access data in hf5 format.
        read method is used to load data from a preprocessed file.
        The file format is a table with each column is a sensor and
        each row an event
    """

    def __init__(self,path,file_name):
        self.path = path
        self.file_name = file_name

    def read(self):
        os.chdir(self.path)
        self.data = pd.read_hdf(self.file_name,key='MC')
        self.tdc  = pd.read_hdf(self.file_name,key='TDC')
        # Reads translated hf files (table with sensor/charge per event)
        self.sensors = np.array(self.data.columns)
        self.data = np.array(self.data, dtype = 'int32')
        self.tdc  = np.array(self.tdc , dtype = 'int32')
        self.events = self.data.shape[0]

        #returns data array, sensors vector, and number of events
        return self.data,self.tdc,self.sensors,self.events

    def read_DAQ_fast(self):
        file_name = self.path+self.file_name
        with pd.HDFStore(file_name) as hdf:
            out={}
            for i in hdf.keys():
                out[i] = pd.read_hdf(file_name,key=i)
        return out


class hdf_compose(object):
    """ A utility class to access preprocessed data from MCs in hf5 format.
            param
            files           : Array of files
            n_sensors       : Number of sensors (all of them)
            Output
            composed data
            sensor array
            number of events
    """

    def __init__(self,path,file_name,files,n_sensors):
        self.path       = path
        self.file_name  = file_name
        self.files      = files
        self.n_sensors  = n_sensors
        self.data       = np.array([]).reshape(0,self.n_sensors)
        self.data_aux   = np.array([]).reshape(0,self.n_sensors)
        self.tdc        = np.array([]).reshape(0,self.n_sensors)
        self.tdc_aux    = np.array([]).reshape(0,self.n_sensors)

    def compose(self):

        hf = hdf_access(self.path,self.file_name + str(self.files[0]).zfill(3) + ".h5")
        self.data_aux,self.tdc_aux,self.sensors,self.events = hf.read()

        self.data = np.pad( self.data,
                            ((0,self.events),(0,0)),
                            mode='constant',
                            constant_values=0)
        self.data[-self.events:,:] = self.data_aux

        self.tdc = np.pad( self.tdc,
                            ((0,self.events),(0,0)),
                            mode='constant',
                            constant_values=0)
        self.tdc[-self.events:,:] = self.tdc_aux

        for i in self.files:
            hf = hdf_access(self.path,self.file_name + str(i).zfill(3) + ".h5")
            self.data_aux, self.tdc_aux, self.fake, self.events = hf.read()

            self.data = np.pad( self.data,
                                ((0,self.events),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.data[-self.events:,:] = self.data_aux

            self.tdc = np.pad( self.tdc,
                                ((0,self.events),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.tdc[-self.events:,:] = self.tdc_aux

        return self.data, self.tdc, self.sensors, self.data.shape[0]


class CUBE_graphs(object):
    """ Data Analysis and Graphs generation
    """
    def __init__(self,config_file,data_path):
        self.config_file = config_file
        self.data_path   = data_path

    def __call__(self):

        # Read first config_file to get n_L1 (same for all files)
        config_file = self.data_path + self.config_file[0] + ".json"
        CG   = CFG.SIM_DATA(filename = config_file,read = True)
        CG   = CG.data
        n_L1 = np.array(CG['L1']['L1_mapping_O']).shape[0]
        n_sipms_int = CG['TOPOLOGY']['sipm_int_row']*CG['TOPOLOGY']['n_rows']
        n_sipms_ext = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
        n_sipms     = n_sipms_int + n_sipms_ext

        log_ETHOUT   = np.array([]).reshape(0,2)
        log_FIFOIN   = np.array([]).reshape(0,2)
        log_asicout  = np.array([]).reshape(0,2)
        log_channels = np.array([]).reshape(0,2)

        lost_ETHOUT   = np.array([]).reshape(0,1)
        lost_FIFOIN   = np.array([]).reshape(0,1)
        lost_asicout  = np.array([]).reshape(0,1)
        lost_channels = np.array([]).reshape(0,1)
        lost_producers = np.array([]).reshape(0,1)

        in_time      = np.array([]).reshape(0,n_sipms)
        out_time     = np.array([]).reshape(0,n_sipms)


        for i in self.config_file:

            start = i.rfind("/")
            jsonname = i[start+1:]

            config_file2 = self.data_path + i + ".json"
            CG = CFG.SIM_DATA(filename = config_file2,read = True)
            CG = CG.data
            chain = CG['ENVIRONMENT']['out_file_name'][CG['ENVIRONMENT']['out_file_name'].rfind("./")+1:]
            filename = chain + "_" + jsonname + ".h5"
            filename = self.data_path + filename


            log_ETHOUT   = np.vstack([log_ETHOUT,np.array(pd.read_hdf(filename,key='log_ETHOUT'))])
            log_FIFOIN   = np.vstack([log_FIFOIN,np.array(pd.read_hdf(filename,key='log_FIFOIN'))])
            log_asicout  = np.vstack([log_asicout,np.array(pd.read_hdf(filename,key='log_asicout'))])
            log_channels = np.vstack([log_channels,np.array(pd.read_hdf(filename,key='log_channels'))])

            lost_ETHOUT   = np.vstack([lost_ETHOUT,np.array(pd.read_hdf(filename,key='lost_ETHOUT'))])
            lost_FIFOIN   = np.vstack([lost_FIFOIN,np.array(pd.read_hdf(filename,key='lost_FIFOIN'))])
            lost_asicout  = np.vstack([lost_asicout,np.array(pd.read_hdf(filename,key='lost_asicout'))])
            lost_channels = np.vstack([lost_channels,np.array(pd.read_hdf(filename,key='lost_channels'))])
            lost_producers = np.vstack([lost_producers,np.array(pd.read_hdf(filename,key='lost_producers'))])

            in_time      = np.vstack([in_time,np.array(pd.read_hdf(filename,key='in_time'))])
            out_time     = np.vstack([out_time,np.array(pd.read_hdf(filename,key='out_time'))])


        # Parameter computation
        latency    = np.max((out_time-in_time)/1E6,axis=1)

        # CRT
        ring_length = CG['TOPOLOGY']['sipm_ext_row']
        # Mapping Function
        try:
            style = CG['L1']['map_style']
            L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(CG, style)
        except:
            # JSON file doesn't include mapping option
            L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = MAP.SiPM_Mapping(CG, 'striped')

        ring_dim     = SiPM_Matrix_O.shape
        in_time_M    = np.ma.MaskedArray(in_time,in_time<1)
        gamma1_sipm  = np.ma.argmin(in_time_M,axis=1)
        gamma1_tdc   = np.ma.filled(np.ma.min(in_time_M,axis=1))
        print gamma1_sipm
        gamma2_sipm  = np.zeros(gamma1_sipm.shape)
        gamma2_tdc   = np.zeros(gamma1_sipm.shape)

        for i in range(CG['ENVIRONMENT']['n_events']):
            gamma1_coord = np.where(SiPM_Matrix_O==gamma1_sipm[i])
            # Roll SiPM Matrixes to find opposite side of detector
            Xe = np.roll(SiPM_Matrix_O,-gamma1_coord[1]+ring_dim[1]//4,axis=1)
            # Select opposite side of detector
            Xe_sel = Xe[:,ring_dim[1]//2:]
            Xe_sel_1D = Xe_sel.reshape(-1)
            try:
                gamma2_tdc[i] = np.ma.min(in_time_M[i,Xe_sel_1D])
            except:
                gamma2_tdc[i] = 0

            # print "gamma1_coord = ",gamma1_coord[1]
            # print "gamma1_tdc = ",gamma1_tdc[i]
            # print "gamma2_tdc = ",gamma2_tdc[i]
            # print "TOF = ",gamma1_tdc[i]-gamma2_tdc[i]
            #gamma2_sipm[i] = Xe_sel_1D[np.argmin(in_time[i,Xe_sel_1D])]

        # Get rid of singles
        TOF = (gamma1_tdc - gamma2_tdc)/2
        TOF = TOF[np.logical_not(np.isnan(TOF))]
        print TOF.shape
        # Introduce a random sign to symmetrize distribution
        random_sign = (np.random.rand(TOF.shape[0])>0.5)*-1
        random_sign = random_sign + random_sign + 1
        TOF = TOF * random_sign


        print ("LOST DATA PRODUCER -> CH      = %d" % (lost_producers.sum()))
        print ("LOST DATA CHANNELS -> OUTLINK = %d" % (lost_channels.sum()))
        print ("LOST DATA OUTLINK  -> L1      = %d" % (lost_asicout.sum()))
        print ("LOST DATA L1A -> L1B          = %d" % (lost_FIFOIN.sum()))
        print ("LOST DATA L1 Ethernet Link    = %d" % (lost_ETHOUT.sum()))

        WC_CH_FIFO    = float(max(log_channels[:,0])/CG['TOFPET']['IN_FIFO_depth'])*100
        WC_OLINK_FIFO = float(max(log_asicout[:,0])/CG['TOFPET']['OUT_FIFO_depth'])*100
        WC_L1_A_FIFO  = float(max(log_FIFOIN[:,0])/CG['L1']['FIFO_L1a_depth'])*100
        WC_L1_B_FIFO  = float(max(log_ETHOUT[:,0])/CG['L1']['FIFO_L1b_depth'])*100


        print ("\n \n BYE \n \n")

        fit = fit_library.gauss_fit()
        fig = plt.figure(figsize=(15,10))

        pos = 431
        fit(log_channels[:,0],range(1,CG['TOFPET']['IN_FIFO_depth']+2))
        fit.plot(axis = fig.add_subplot(pos),
                title = "ASICS Channel Input analog FIFO (4)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(pos).set_yscale('log')
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.add_subplot(pos).text(0.99,0.97,(("ASIC Input FIFO reached %.1f %%" % \
                                                (WC_CH_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)

        pos = 432
        fit(log_asicout[:,0],CG['TOFPET']['OUT_FIFO_depth']/10)
        fit.plot(axis = fig.add_subplot(pos),
                title = "ASICS Channels -> Outlink",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(pos).set_yscale('log')
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.add_subplot(pos).text(0.99,0.97,(("ASIC Outlink FIFO reached %.1f %%" % \
                                                (WC_OLINK_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)

        pos = 434
        fit(log_FIFOIN[:,0],CG['L1']['FIFO_L1a_depth']/10)
        fit.plot(axis = fig.add_subplot(pos),
                title = "ASICS -> L1A (FIFOIN)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(pos).set_yscale('log')
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.add_subplot(pos).text(0.99,0.97,(("L1_A FIFO reached %.1f %%" % \
                                                (WC_L1_A_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))


        pos = 435
        fit(log_ETHOUT[:,0],CG['L1']['FIFO_L1b_depth']/10)
        fit.plot(axis = fig.add_subplot(pos),
                title = "L1 OUTPUT (ETHOUT)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(pos).set_yscale('log')
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.add_subplot(pos).text(0.99,0.97,(("L1_B FIFO reached %.1f %%" % \
                                                (WC_L1_B_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))

        pos = 436
        fit(latency,80)
        fit.plot(axis = fig.add_subplot(pos),
                title = "Total Data Latency",
                xlabel = "Latency in microseconds",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(pos).text(0.99,0.8,(("WORST LATENCY = %f us" % \
                                                (max(latency)))),
                                                fontsize=7,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)
        fig.add_subplot(pos).text(0.99,0.7,(("MEAN LATENCY = %f us" % \
                                                (np.mean(latency)))),
                                                fontsize=7,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(pos).transAxes)
        fig.add_subplot(pos).xaxis.set_major_locator(MaxNLocator(integer=True))


        pos = 433
        new_axis = fig.add_subplot(pos)
        new_axis.text(0.05,0.8,(("LOST DATA PRODUCER -> CH           = %d\n" + \
                                 "LOST DATA CHANNELS -> OUTLINK  = %d\n" + \
                                 "LOST DATA OUTLINK -> L1                = %d\n" + \
                                 "LOST DATA L1A -> L1B                      = %d\n" + \
                                 "LOST ETHERLINK                      = %d\n") % \
                                (lost_producers.sum(),
                                 lost_channels.sum(),
                                 lost_asicout.sum(),
                                 lost_FIFOIN.sum(),
                                 lost_ETHOUT.sum())
                                ),
                                fontsize=8,
                                verticalalignment='top',
                                horizontalalignment='left',
                                transform=new_axis.transAxes)

        pos = 437
        range_tof_h = TOF<2000
        range_tof_l = TOF>-2000
        range_tof = range_tof_l * range_tof_h
        fit(TOF[range_tof],100)
        fit.plot(axis = fig.add_subplot(pos),
                title = "Time of Flight ",
                xlabel = "Time Stamp in picoseconds",
                ylabel = "Hits",
                res = False, fit =True)

        new_axis.text(0.05,0.9,(("Time of Flight Resolution = %d\n") % \
                                (lost_producers.sum())),
                                fontsize=8,
                                verticalalignment='top',
                                horizontalalignment='left',
                                transform=new_axis.transAxes)


        fig.tight_layout()
        plt.savefig(filename + ".pdf")

if __name__ == "__main__":
    main()
