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
        self.data_np  = np.array([]).reshape(0,6)

        self.n_L1       = np.array(CG['L1']['L1_mapping_O']).shape[0]
        self.n_asics    = np.sum(np.array(CG['L1']['L1_mapping_O']))
        self.n_sipm     = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
        self.first_sipm = CG['TOPOLOGY']['first_sipm']
        self.n_events    = CG['ENVIRONMENT']['n_events']


    def write_raw_out(self):

        for i in range(len(self.pool_out)):
            for j in self.pool_out[i]['DATA_out']:
                data = np.array(j.values())
                self.data_np = np.vstack([self.data_np,data])

        topo_data = np.array(list(self.topology.values())).reshape(1,len(list(self.topology.values())))

        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile_raw,
                        complevel=9, complib='zlib') as store:

            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            raw_data      = pd.DataFrame(data = self.data_np,
                                         columns=self.pool_out[0]['DATA_out'][0].keys())
            topo          = pd.DataFrame(data = topo_data,
                                         columns = list(self.topology.keys()))

            store.put('raw',raw_data)
            store.put('sensors',sensors_array)
            store.put('topology',topo)
            store.close()


    def process(self):

        """ POOL_OUT = Simulation Pool Output
            {'DATA_out' : ch_frame array,
             'L1_out'   : L1 logs,
             'ASICS_out': ASICS logs}
        """

        # Translate Simulation Output into an array for Data recovery
        SIM_OUT = {'DATA_out':[],'L1_out':[], 'ASICS_out':[]}
        for i in range(len(self.pool_out)):
            SIM_OUT['L1_out'].extend(self.pool_out[i]['L1_out'])
            SIM_OUT['DATA_out'].extend(self.pool_out[i]['DATA_out'])
            for j in range(len(self.pool_out[i]['ASICS_out'])):
                SIM_OUT['ASICS_out'].extend(self.pool_out[i]['ASICS_out'][j])


        lost_producers  = np.array([]).reshape(0,1)
        lost_channels   = np.array([]).reshape(0,1)
        lost_outlink    = np.array([]).reshape(0,1)
        log_channels    = np.array([]).reshape(0,2)
        log_outlink     = np.array([]).reshape(0,2)

        SIM_OUT_L1      = np.array(SIM_OUT['L1_out'])
        SIM_OUT_ASICs   = np.array(SIM_OUT['ASICS_out'])
        lost_FIFOIN     = np.array([]).reshape(0,1)
        lost_ETHOUT     = np.array([]).reshape(0,1)
        log_FIFOIN      = np.array([]).reshape(0,2)
        log_ETHOUT      = np.array([]).reshape(0,2)


        # Gather Log information from ASICS layer
        for j in range(self.n_asics):
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

        # Gather Log information from L1 layer
        for j in range(self.n_L1):
            lost_FIFOIN = np.vstack([lost_FIFOIN,
                                        SIM_OUT_L1[j]['lost_FIFOIN']])
            lost_ETHOUT = np.vstack([lost_ETHOUT,
                                        SIM_OUT_L1[j]['lost_ETHOUT']])
            log_FIFOIN = np.vstack([log_FIFOIN,
                                        SIM_OUT_L1[j]['log_FIFOIN']])
            log_ETHOUT = np.vstack([log_ETHOUT,
                                        SIM_OUT_L1[j]['log_ETHOUT']])


        # data_sit: list of ch_frame dictionaries sorted by in time (sit)
        data_sit = sorted(SIM_OUT['DATA_out'], key=lambda k:k['in_time'])

        data_panel = np.zeros([self.n_events,self.n_sipm],dtype=int)

        # Llenar la tabla
        index = 0
        count = 0
        for i in range(self.n_events):
            timestamp = data_sit[index]['in_time']
            while timestamp ==  data_sit[count]['in_time']:
                data_panel[i][sensor_id] = data_sit[count]['data']
                count = count + 1
            index = count



        output = {'QDC_table': data_panel,

                  'L1_log': {'lost_FIFOIN': lost_FIFOIN,
                             'lost_ETHOUT': lost_ETHOUT,
                             'log_FIFOIN' : log_FIFOIN,
                             'log_ETHOUT' : log_ETHOUT},

                  'ASICS_log': { 'lost_producers':lost_producers,
                                 'lost_channels':lost_channels,
                                 'lost_outlink':lost_outlink,
                                 'log_channels':log_channels,
                                 'log_outlink':log_outlink}
                }


        return output


    def write_out(self, p_out):

        topo_data = np.array(list(self.topology.values())).reshape(1,len(list(topology.values())))

        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile,
                        complevel=9, complib='zlib') as store:

            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            qdc_data      = pd.DataFrame(data = np.array(p_out['QDC_table']),
                                         columns=self.sensors_xyz[:,0])
            topo          = pd.DataFrame(data = topo_data,
                                         columns = list(topology.keys()))

            lost_producers = pd.DataFrame(data = np.array(p_out['ASICS_log']['lost_producers']))
            lost_channels  = pd.DataFrame(data = np.array(p_out['ASICS_log']['lost_channels']))
            lost_asicout   = pd.DataFrame(data = np.array(p_out['ASICS_log']['lost_outlink']))
            log_channels   = pd.DataFrame(data = np.array(p_out['ASICS_log']['log_channels']))
            log_asicout    = pd.DataFrame(data = np.array(p_out['ASICS_log']['log_outlink']))

            lost_FIFOIN   = pd.DataFrame(data = np.array(p_out['L1_log']['lost_FIFOIN']))
            lost_ETHOUT   = pd.DataFrame(data = np.array(p_out['L1_log']['lost_ETHOUT']))
            log_FIFOIN    = pd.DataFrame(data = np.array(p_out['L1_log']['log_FIFOIN']))
            log_ETHOUT    = pd.DataFrame(data = np.array(p_out['L1_log']['log_ETHOUT']))

            store.put('qdc_data',raw_data)
            store.put('sensors',sensors_array)
            store.put('topology',topo)

            store.put('lost_producers',lost_producers)
            store.put('lost_channels',lost_channels)
            store.put('lost_asicout',lost_asicout)
            store.put('lost_FIFOIN',lost_FIFOIN)
            store.put('lost_ETHOUT',lost_ETHOUT)

            store.put('log_channels',log_channels)
            store.put('log_asicout',log_channels)
            store.put('log_FIFOIN',log_channels)
            store.put('log_ETHOUT',log_channels)
            store.close()


class DAQ_IO(object):
    """ H5 access for DAQ.
        Methods
        write       : Writes H5 with output dataframes from DAQ (see DATA FRAME)
                      Also attaches a table with sensor positions
        read        : Reads DAQ output frames from H5 file
        write_out   : Writes processed output frames. The output format is the
                      same of the input processed H5 (Table with Sensors as columns
                      and events as rows)
        Parameters

    """

    def __init__(self,path,daq_filename,ref_filename,daq_outfile):
        self.path = path
        self.out_filename = daq_filename
        self.ref_filename = ref_filename
        self.daq_outfile = daq_outfile
        os.chdir(self.path)
        self.sensors_xyz = np.array( pd.read_hdf(self.ref_filename,
                                    key='sensors'),
                                    dtype = 'float32')

    def write(self,data):
        os.chdir(self.path)
        with pd.HDFStore(self.out_filename) as store:
            panel_array = pd.DataFrame( data = data,
                                        columns = ['data','event','sensor_id',
                                                    'asic_id','in_time','out_time'])
            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',panel_array)
            store.put('sensors',sensors_array)
            store.close()


    # def read(self):
    #     os.chdir(self.path)
    #     data = np.array(pd.read_hdf(self.out_filename,key='MC'), dtype='int32')
    #     sensors = np.array(np.array(pd.read_hdf(self.out_filename,key='sensors'),
    #               dtype='int32'))
    #     return data,sensors

    def read(self):
        data = np.array(pd.read_hdf(self.path+self.out_filename,
                                    key='MC'), dtype='int32')
        sensors = np.array(np.array(pd.read_hdf(self.path+self.out_filename,
                                    key='sensors'),dtype='int32'))
        return data,sensors


    def write_out(self,data,topology={},logs={}):
        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile,
                        complevel=9, complib='zlib') as store:
            self.panel_array = pd.DataFrame( data=data,
                                             columns=self.sensors_xyz[:,0])

            self.sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                                columns=['sensor','x','y','z'])
            topo_data = np.array(list(topology.values())).reshape(1,len(list(topology.values())))
            logA         = np.array(logs['logA'])
            logB         = np.array(logs['logB'])
            logC         = np.array(logs['logC'])
            frame_frag   = logs['frame_frag']
            log_channels = np.array(logs['log_channels'])
            log_outlink = np.array(logs['log_outlink'])
            log_in_time = np.array(logs['in_time'])
            log_out_time = np.array(logs['out_time'])

            topo = pd.DataFrame(data = topo_data,columns = list(topology.keys()))
            logA = pd.DataFrame(data = logA)
            logB = pd.DataFrame(data = logB)
            logC = pd.DataFrame(data = logC)
            frame_frag = pd.DataFrame(data = frame_frag)
            log_channels = pd.DataFrame(data = log_channels)
            log_outlink  = pd.DataFrame(data = log_outlink)
            log_in_time  = pd.DataFrame(data = log_in_time)
            log_out_time  = pd.DataFrame(data = log_out_time)
            lost_data = np.array(list(logs['lost'].values())).reshape(1,-1)
            lost = pd.DataFrame(data = lost_data,columns = list(logs['lost'].keys()))
            compress = pd.DataFrame(data = logs['compress'])
            tstamp_event = pd.DataFrame(data = logs['tstamp_event'])
            timestamp = pd.DataFrame(data=logs['timestamp'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',self.panel_array)
            store.put('sensors',self.sensors_array)
            store.put('topology',topo)
            store.put('logA',logA)
            store.put('logB',logB)
            store.put('logC',logC)
            store.put('log_channels',log_channels)
            store.put('log_outlink',log_outlink)
            store.put('in_time',log_in_time)
            store.put('out_time',log_out_time)
            store.put('lost',lost)
            store.put('compress',compress)
            store.put('tstamp_event',tstamp_event)
            store.put('timestamp',timestamp)
            store.put('frame_frag',frame_frag)
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

        # Reads translated hf files (table with sensor/charge per event)
        self.sensors = np.array(self.data.columns)
        self.data = np.array(self.data, dtype = 'int32')
        self.events = self.data.shape[0]

        #returns data array, sensors vector, and number of events
        return self.data,self.sensors,self.events

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

    def compose(self):

        hf = hdf_access(self.path,self.file_name + str(self.files[0]).zfill(3) + ".h5")
        self.data_aux,self.sensors,self.events = hf.read()
        self.data = np.pad( self.data,
                            ((0,self.events),(0,0)),
                            mode='constant',
                            constant_values=0)
        self.data[-self.events:,:] = self.data_aux

        for i in self.files:
            hf = hdf_access(self.path,self.file_name + str(i).zfill(3) + ".h5")
            self.data_aux,self.fake,self.events = hf.read()
            self.data = np.pad( self.data,
                                ((0,self.events),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.data[-self.events:,:] = self.data_aux


        return self.data, self.sensors, self.data.shape[0]


# class infinity_graphs(object):
#     """ Data Analysis and Graphs generation
#     """
#     def __init__(self,config_file,data_path):
#         self.config_file = config_file
#         self.data_path   = data_path
#
#     def __call__(self):
#
#         # Read first config_file to get n_L1 (same for all files)
#         config_file = self.data_path + self.config_file[0] + ".json"
#         CG   = CFG.SIM_DATA(filename = config_file,read = True)
#         CG   = CG.data
#         n_L1 = np.array(CG['L1']['L1_mapping_O']).shape[0]
#
#         logA         = np.array([]).reshape(0,2)
#         logB         = np.array([]).reshape(0,2)
#         logC         = np.array([]).reshape(0,2)
#         log_channels = np.array([]).reshape(0,2)
#         log_outlink  = np.array([]).reshape(0,2)
#         in_time      = np.array([]).reshape(0,1)
#         out_time     = np.array([]).reshape(0,1)
#         lost         = np.array([]).reshape(0,4)
#         compress     = np.array([]).reshape(0,1)
#         frame_frag   = np.array([]).reshape(0,n_L1+1)
#
#         for i in self.config_file:
#             # jsonname = string.replace(i,"/","_")
#             # jsonname = string.replace(jsonname,".","")
#             start = i.rfind("/")
#             jsonname = i[start+1:]
#
#             config_file2 = self.data_path + i + ".json"
#             CG = CFG.SIM_DATA(filename = config_file2,read = True)
#             CG = CG.data
#             chain = CG['ENVIRONMENT']['out_file_name'][CG['ENVIRONMENT']['out_file_name'].rfind("./")+2:]
#             filename = chain + "_" + jsonname + ".h5"
#             filename = self.data_path + filename
#
#             logA         = np.vstack([logA,np.array(pd.read_hdf(filename,key='logA'))])
#             logB         = np.vstack([logB,np.array(pd.read_hdf(filename,key='logB'))])
#             logC         = np.vstack([logC,np.array(pd.read_hdf(filename,key='logC'))])
#             log_channels = np.vstack([log_channels,np.array(pd.read_hdf(filename,key='log_channels'))])
#             log_outlink  = np.vstack([log_outlink,np.array(pd.read_hdf(filename,key='log_outlink'))])
#             in_time      = np.vstack([in_time,np.array(pd.read_hdf(filename,key='in_time'))])
#             out_time     = np.vstack([out_time,np.array(pd.read_hdf(filename,key='out_time'))])
#             lost         = np.vstack([lost,np.array(pd.read_hdf(filename,key='lost'))])
#             compress     = np.vstack([compress,np.array(pd.read_hdf(filename,key='compress'))])
#             frame_frag   = np.vstack([frame_frag,np.array(pd.read_hdf(filename,key='frame_frag'))])
#
#         latency_L1 = logA[:,1]
#         latency    = out_time-in_time
#
#         print ("LOST DATA PRODUCER -> CH      = %d" % (lost[:,0].sum()))
#         print ("LOST DATA CHANNELS -> OUTLINK = %d" % (lost[:,1].sum()))
#         print ("LOST DATA OUTLINK  -> L1      = %d" % (lost[:,2].sum()))
#         print ("LOST DATA L1A -> L1B          = %d" % (lost[:,3].sum()))
#
#         WC_CH_FIFO    = float(max(log_channels[:,0])/CG['TOFPET']['IN_FIFO_depth'])*100
#         WC_OLINK_FIFO = float(max(log_outlink[:,0])/CG['TOFPET']['OUT_FIFO_depth'])*100
#         WC_L1_A_FIFO  = float(max(logA[:,0])/CG['L1']['FIFO_L1a_depth'])*100
#         WC_L1_B_FIFO  = float(max(logB[:,0])/CG['L1']['FIFO_L1b_depth'])*100
#
#
#         print ("\n \n BYE \n \n")
#
#         fit = fit_library.gauss_fit()
#         fig = plt.figure(figsize=(20,10))
#
#         fit(log_channels[:,0],range(1,CG['TOFPET']['IN_FIFO_depth']+2))
#         fit.plot(axis = fig.add_subplot(341),
#                 title = "ASICS Channel Input analog FIFO (4)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(341).set_yscale('log')
#         fig.add_subplot(341).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(341).text(0.99,0.97,(("ASIC Input FIFO reached %.1f %%" % \
#                                                 (WC_CH_FIFO))),
#                                                 fontsize=8,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(341).transAxes)
#
#         fit(log_outlink[:,0],CG['TOFPET']['OUT_FIFO_depth'])
#         fit.plot(axis = fig.add_subplot(342),
#                 title = "ASICS Channels -> Outlink",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(342).set_yscale('log')
#         fig.add_subplot(342).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(342).text(0.99,0.97,(("ASIC Outlink FIFO reached %.1f %%" % \
#                                                 (WC_OLINK_FIFO))),
#                                                 fontsize=8,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(342).transAxes)
#
#         fit(logA[:,0],CG['L1']['FIFO_L1a_depth'])
#         fit.plot(axis = fig.add_subplot(346),
#                 title = "ASICS -> L1A (FIFOA)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(346).set_yscale('log')
#         fig.add_subplot(346).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(346).text(0.99,0.97,(("L1_A FIFO reached %.1f %%" % \
#                                                 (WC_L1_A_FIFO))),
#                                                 fontsize=8,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(346).transAxes)
#         fig.add_subplot(3,4,6).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         fit(logB[:,0],CG['L1']['FIFO_L1b_depth'])
#         fit.plot(axis = fig.add_subplot(345),
#                 title = "L1 OUTPUT (FIFOB)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(345).set_yscale('log')
#         fig.add_subplot(345).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(345).text(0.99,0.97,(("L1_B FIFO reached %.1f %%" % \
#                                                 (WC_L1_B_FIFO))),
#                                                 fontsize=8,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(345).transAxes)
#         fig.add_subplot(3,4,5).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         fit(logC[:,0],range(int(np.max(logC[:,0]))+2))
#         fit.plot(axis = fig.add_subplot(3,4,10),
#                 title = "Number of Frames per Buffer",
#                 xlabel = "Number of Frames",
#                 ylabel = "Hits",
#                 res = False, fit = True)
#         fig.add_subplot(3,4,10).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         fit(latency,50)
#         fit.plot(axis = fig.add_subplot(343),
#                 title = "Total Data Latency",
#                 xlabel = "Latency in nanoseconds",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(343).text(0.99,0.8,(("WORST LATENCY = %d ns" % \
#                                                 (max(latency)))),
#                                                 fontsize=7,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(343).transAxes)
#         fig.add_subplot(343).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         new_axis = fig.add_subplot(347)
#         x_data = fit.bin_centers
#         y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
#         new_axis.plot(x_data,y_data)
#         new_axis.set_ylim((0.9,1.0))
#         new_axis.set_xlabel("Latency in nanoseconds")
#         new_axis.set_ylabel("Percentage of Recovered Data")
#         new_axis.text(0.05,0.9,(("LOST DATA PRODUCER -> CH           = %d\n" + \
#                                  "LOST DATA CHANNELS -> OUTLINK  = %d\n" + \
#                                  "LOST DATA OUTLINK -> L1                = %d\n" + \
#                                  "LOST DATA L1A -> L1B                      = %d\n") % \
#                                 (lost[:,0].sum(),
#                                  lost[:,1].sum(),
#                                  lost[:,2].sum().sum(),
#                                  lost[:,3].sum().sum())
#                                 ),
#                                 fontsize=8,
#                                 verticalalignment='top',
#                                 horizontalalignment='left',
#                                 transform=new_axis.transAxes)
#
#         fit(latency_L1,50)
#         fit.plot(axis = fig.add_subplot(349),
#                 title = "L1 input Data Latency",
#                 xlabel = "Latency in nanoseconds",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(349).text(0.99,0.8,(("WORST LATENCY = %d ns" % \
#                                                 (max(latency_L1)))),
#                                                 fontsize=7,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(349).transAxes)
#         fig.add_subplot(349).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         fit(compress,int(np.max(compress)))
#         fit.plot(axis = fig.add_subplot(344),
#                 title = "Data Frame Length",
#                 xlabel = "Number of QDC fields",
#                 ylabel = "Hits",
#                 res = False,
#                 fit = False)
#         fig.add_subplot(344).set_yscale('log')
#         fig.add_subplot(344).xaxis.set_major_locator(MaxNLocator(integer=True))
#         # TOTAL NUMBER OF BITS vs COMPRESS EFFICIENCY
#         A = np.arange(0,np.max(compress))
#         D_data = [DAQ.L1_outframe_nbits(i) for i in A]
#         #D_data = 1 + 7*(A>0) + A * 23 + 10     #see DAQ_infinity
#         D_save = (A-1)*10
#         #This is what you save when only one TDC is sent
#
#         B_data = np.multiply(D_data,fit.hist)
#         B_save = np.multiply(D_save,fit.hist)
#         B_save[0]=0
#         B_save[1]=0
#         new_axis_2 = fig.add_subplot(348)
#         x_data = fit.bin_centers
#         new_axis_2.bar(x_data,B_data,color='r')
#         new_axis_2.bar(x_data,B_save,color='b')
#         new_axis_2.set_title("Data sent vs frame length")
#         new_axis_2.set_xlabel("Length of frame in QDC data")
#         new_axis_2.set_ylabel("Red - Data sent (bits) / Blue - Data saved (bits)")
#
#         new_axis_2.text(0.99,0.97,(("TOTAL DATA SENT = %d bits\n" + \
#                                  "DATA REDUCTION  = %d bits\n" + \
#                                  "COMPRESS RATIO = %f \n") % \
#                                 (np.sum(B_data),np.sum(B_save),float(np.sum(B_data))/float(np.sum(B_save)+np.sum(B_data)))),
#                                 fontsize=8,
#                                 verticalalignment='top',
#                                 horizontalalignment='right',
#                                 transform=new_axis_2.transAxes)
#
#
#         ############### FRAME FRAGMENTATION ANALYSIS ##########################
#         # TIME FRAGMENTATION
#         frag_matrix = frame_frag[1:,1:]
#         frag_matrix = frag_matrix.reshape(-1)
#         frag_matrix = (frag_matrix>0)*frag_matrix
#         fit(frag_matrix,range(1,int(np.max(frag_matrix))+2))
#         print int(np.max(frag_matrix))
#         fit.plot(axis = fig.add_subplot(3,4,11),
#                 title = "Frame Fragmentation - (TIME)",
#                 xlabel = "Frame pieces (Buffers)",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(3,4,11).set_yscale('log')
#         fig.add_subplot(3,4,11).set_ylim(bottom=1)
#         fig.add_subplot(3,4,11).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         # SPATIAL FRAGMENTATION
#         cluster_record = []
#         frag_matrix = frame_frag[:,1:]
#
#         for ev in frag_matrix:
#             clusters_aux = np.zeros((1,2))
#             clusters = np.array([]).reshape(0,2)
#             flag = False
#             L1_count = 0
#             # First element , Last element
#             for L1 in ev:
#                 if (L1 > 0):
#                     # Cluster detected
#                     if (flag == False):
#                         # First element in the cluster
#                         clusters_aux[0,0] = L1_count
#                         clusters_aux[0,1] = L1_count
#                         flag = True
#                     else:
#                         # Inside the cluster
#                         clusters_aux[0,1] = L1_count
#                         flag = True
#
#                     if (L1_count == (n_L1-1)):
#                         clusters = np.vstack([clusters,clusters_aux])
#                 else:
#                     if (flag == True):
#                         # End of cluster
#                         flag = False
#                         clusters = np.vstack([clusters,clusters_aux])
#                         clusters_aux = np.zeros((1,2))
#                     else:
#                         # No cluster detected
#                         flag = False
#
#                 L1_count += 1
#
#             # Must solve special case of boundary cluster in circular buffer
#             if ((clusters[0,0] == 0) and (clusters[-1,1] == (L1_count-1))):
#                 clusters[0,0] = clusters[-1,0]
#                 clusters = clusters[:-1,:]
#
#             cluster_record.append(clusters)
#
#         cluster_lengths = []
#         # LET'S FIND CLUSTER LENGTHS
#         for i in cluster_record:
#             for j in i:
#                 if (j[1] >= j[0]):
#                     cluster_lengths.append(int(j[1]-j[0]+1))
#                 else:
#                     cluster_lengths.append(int(j[1]-j[0]+n_L1+1))
#
#         fit(cluster_lengths,range(1,int(np.max(cluster_lengths))+2))
#         fit.plot(axis = fig.add_subplot(3,4,12),
#                 title = "Frame Fragmentation - (SPACE)",
#                 xlabel = "Number of L1 per event",
#                 ylabel = "Hits",
#                 res = False,
#                 fit = False)
#         fig.add_subplot(3,4,12).set_yscale('log')
#         fig.add_subplot(3,4,12).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         fig.tight_layout()
#
#         #plt.savefig(CG['ENVIRONMENT']['out_file_name']+"_"+ filename + ".pdf")
#         plt.savefig(filename + ".pdf")
#
#
# class encoder_graphs(object):
#     def __init__(self,config_file,data_path):
#         self.config_file = config_file
#         self.data_path   = data_path
#         self.sipm_polar = 0
#
#     def measure(self,event,**kargs):
#
#         roi_size    = kargs['roi_size']   # = 32
#         roi_height  = kargs['roi_height'] # = 16
#         offset      = kargs['offset']     # int(self.sipm_polar[0,0])
#         radius      = kargs['radius']     # 161
#         SiPM_Matrix = kargs['SiPM_Matrix'] # with OFFSET !!!!
#         d_TE        = kargs['d_TE']
#         d_recons    = kargs['d_recons']
#
#
#         max_sipm = np.argwhere(d_TE[event,:] == np.max(d_TE[event,:]))+offset
#         x_y  = np.argwhere(SiPM_Matrix == max_sipm[0])
#
#         # Build ROIs for subevents
#         roi_matrix1  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1],axis=1)[:,0:roi_size].astype(int)
#         roi_matrix2  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1]-SiPM_Matrix.shape[1]//2,axis=1)[:,0:roi_size].astype(int)
#
#         # Check if there is something on the opposite side
#         #if (np.max(d_TE[event,roi_matrix2-offset]) == 0):
#         #    roi_matrix2 = np.array([[]])
#
#         roi_center1 = roi_matrix1[ roi_height//2, roi_size//2 ]
#         roi_center2 = roi_matrix2[ roi_height//2, roi_size//2 ]
#
#         # Get subevent values
#         subevent1_o = d_TE[event,roi_matrix1-offset]
#         subevent1_r = d_recons[event,roi_matrix1-offset]
#         if (roi_matrix2.size > 0):
#             subevent2_o = d_TE[event,roi_matrix2-offset]
#             subevent2_r = d_recons[event,roi_matrix2-offset]
#         else:
#             subevent2_o = np.array([[]])
#             subevent2_r = np.array([[]])
#
#         # Measure parameters
#         # No need to normalize z
#         z_mean1_o =np.mean(subevent1_o*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_o =np.mean(subevent2_o*self.sipm_polar[roi_matrix2-offset,2])
#         z_mean1_r =np.mean(subevent1_r*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_r =np.mean(subevent2_r*self.sipm_polar[roi_matrix2-offset,2])
#
#         # Phi must me normalized to avoid -pi,+pi effects
#         phi_mean1_o = np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_o = np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#         phi_mean1_r = np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_r = np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#
#         z_std1_o   = np.sqrt(np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_o)**2))
#         z_std2_o   = np.sqrt(np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_o)**2))
#         z_std1_r   = np.sqrt(np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_r)**2))
#         z_std2_r   = np.sqrt(np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_r)**2))
#
#         phi_std1_o = np.sqrt(np.mean(subevent1_o*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_o)**2)))
#         phi_std2_o = np.sqrt(np.mean(subevent2_o*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_o)**2)))
#         phi_std1_r = np.sqrt(np.mean(subevent1_r*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_r)**2)))
#         phi_std2_r = np.sqrt(np.mean(subevent2_r*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_r)**2)))
#
#         phi_std1_cart_o = phi_std1_o * radius
#         phi_std2_cart_o = phi_std2_o * radius
#         phi_std1_cart_r = phi_std1_r * radius
#         phi_std2_cart_r = phi_std2_r * radius
#
#         sigma_comb1_o = np.sqrt(z_std1_o**2+phi_std1_cart_o**2)
#         sigma_comb2_o = np.sqrt(z_std2_o**2+phi_std2_cart_o**2)
#         sigma_comb1_r = np.sqrt(z_std1_r**2+phi_std1_cart_r**2)
#         sigma_comb2_r = np.sqrt(z_std2_r**2+phi_std2_cart_r**2)
#
#         out_o, out_r = {},{}
#
#         out_o['z_mean1']     = z_mean1_o        ; out_o['z_mean2']     = z_mean2_o
#         out_o['phi_mean1']   = phi_mean1_o      ; out_o['phi_mean2']   = phi_mean2_o
#         out_o['z_std1']      = z_std1_o         ; out_o['z_std2']      = z_std2_o
#         out_o['phi_std1']    = phi_std1_cart_o  ; out_o['phi_std2']    = phi_std2_cart_o
#         out_o['sigma_comb1'] = sigma_comb1_o    ; out_o['sigma_comb2'] = sigma_comb2_o
#
#         out_r['z_mean1']     = z_mean1_r        ; out_r['z_mean2']     = z_mean2_r
#         out_r['phi_mean1']   = phi_mean1_r      ; out_r['phi_mean2']   = phi_mean2_r
#         out_r['z_std1']      = z_std1_r         ; out_r['z_std2']      = z_std2_r
#         out_r['phi_std1']    = phi_std1_cart_r  ; out_r['phi_std2']    = phi_std2_cart_r
#         out_r['sigma_comb1'] = sigma_comb1_r    ; out_r['sigma_comb2'] = sigma_comb2_r
#
#         return out_o, out_r
#
#
#
#     def __call__(self,roi_size,roi_height):
#
#         CONFIG = CFG.SIM_DATA(filename = self.data_path + \
#                                          self.config_file+".json",
#                               read     = True)
#         orig_array=[]
#         enco_array=[]
#         TE_data_array=[]
#         ENC_data_array=[]
#         n_files             = CONFIG.data['ENVIRONMENT']['n_files']
#         path                = CONFIG.data['ENVIRONMENT']['path_to_files']
#         MC_out_file_name    = CONFIG.data['ENVIRONMENT']['MC_out_file_name']
#         radius              = CONFIG.data['TOPOLOGY']['radius_ext']
#
#
#         # Sensor Positions
#         sipm_cart = np.array(pd.read_hdf(path + MC_out_file_name + "." + \
#                                          str(n_files[0]).zfill(3) + ".h5",
#                                          key='sensors'))
#         self.sipm_polar = np.zeros(sipm_cart.shape)
#         # Columns -> [sensor,r,z,phi]
#         self.sipm_polar[:,0] = sipm_cart[:,0]
#         self.sipm_polar[:,1] = np.sqrt(np.square(sipm_cart[:,1])+np.square(sipm_cart[:,2]))
#         self.sipm_polar[:,2] = sipm_cart[:,3]
#         self.sipm_polar[:,3] = np.arctan2(sipm_cart[:,2],sipm_cart[:,1])
#
#         # Get SiPM Mapping
#         L1, I, SiPM_Matrix, topo = SM.SiPM_Mapping(CONFIG.data,CONFIG.data['L1']['map_style'])
#         offset = int(self.sipm_polar[0,0])
#         SiPM_Matrix = SiPM_Matrix + offset
#
#
#         for j in n_files:
#             d_recons  = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_recons'))
#             d_TE      = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_TE'))
#             d_encoded = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded'))
#
#             for i in range(d_TE.shape[0]):
#
#                 original,encoded = self.measure( event       = i,
#                                                 roi_size    = roi_size,
#                                                 roi_height  = roi_height,
#                                                 offset      = offset,
#                                                 radius      = radius,
#                                                 SiPM_Matrix = SiPM_Matrix,
#                                                 d_TE        = d_TE,
#                                                 d_recons    = d_recons)
#
#                 orig_array.append(original)
#                 enco_array.append(encoded)
#
#                 # Data compression statistics
#                 TE_data_array.append(np.sum(d_TE[i,:]>0))
#                 ENC_data_array.append(np.sum(d_encoded[i,:]>0))
#
#
#         # Processing DONE now ERROR computation
#         orig_np = np.array([np.array(list(orig_array[i].values()),dtype=float) for i in range(len(orig_array))])
#         enco_np = np.array([np.array(list(enco_array[i].values()),dtype=float) for i in range(len(enco_array))])
#         #'sigma_comb2'|'sigma_comb1'|'phi_std1'|'phi_std2'|'phi_mean1'|'phi_mean2'|'z_mean1'|'z_mean2'|'z_std2'|'z_std1'
#
#         Error = orig_np-enco_np
#         z_m_err   = np.concatenate((Error[:,6],Error[:,7]))
#         phi_m_err = np.concatenate((Error[:,4],Error[:,5]))
#         phi_s_err = np.concatenate((Error[:,2],Error[:,3]))
#         z_s_err   = np.concatenate((Error[:,8],Error[:,9]))
#
#         z_m_err   = z_m_err[(z_m_err>-10) * (z_m_err<10) * (z_m_err!=0)]
#         phi_m_err = phi_m_err[(phi_m_err>-0.025) * (phi_m_err<0.025) * (phi_m_err!=0)]*radius
#         # Filter to avoid most of "close to pi" errors
#         phi_s_err = phi_s_err[(phi_s_err>-10) * (phi_s_err<10) * (phi_s_err!=0)]
#         z_s_err   = z_s_err[(z_s_err>-10) * (z_s_err<10)* (z_s_err!=0)]
#
#
#         # Now the plotting stuff
#         err_fit = fit_library.GND_fit()
#         g_fit   = fit_library.gauss_fit()
#
#         fig = plt.figure(figsize=(10,10))
#         err_fit(z_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(321),
#                         title = "Z mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         err_fit(phi_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(322),
#                         title = "PHI mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         err_fit(z_s_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(323),
#                         title = "Z sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         err_fit(phi_s_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(324),
#                         title = "PHI sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         # g_fit.plot(axis = fig.add_subplot(325),
#         #                 title = "DATA sent in TE mode (blue) & ENCODER mode (red)",
#         #                 xlabel = "Number of Words",
#         #                 ylabel = "Hits",
#         #                 res = True, fit = True)
#
#         data_sent = fig.add_subplot(325)
#         g_fit(TE_data_array,'sqrt')
#         x_data = g_fit.bin_centers[g_fit.bin_centers<250]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='b')
#
#         g_fit(ENC_data_array,'sqrt')
#         x_data = g_fit.bin_centers[g_fit.bin_centers<250]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='r')
#
#         data_sent.set_title("DATA sent in TE mode & ENCODER mode")
#         data_sent.set_xlabel("Number of Words")
#         data_sent.set_ylabel("Red - ENCODER (words)) / Blue - TE (words)")
#
#         diff_data = np.array(TE_data_array)-np.array(ENC_data_array)
#         g_fit(diff_data[diff_data<250],'sqrt')
#         diff = fig.add_subplot(326)
#         g_fit.plot(axis = diff,
#                         title = "Difference in Data sent in both modes",
#                         xlabel = "Number of Words",
#                         ylabel = "Hits",
#                         res = False, fit = False)
#         diff.text(0.99,0.97,(("DATA SENT in TE MODE  = %d words\n" + \
#                               "DATA SENT in ENCODER MODE = %d words\n" + \
#                               "COMPRESS RATIO = %f \n") % \
#                               (np.sum(TE_data_array),np.sum(ENC_data_array),
#                                float(np.sum(ENC_data_array))/float(np.sum(TE_data_array)))),
#                                 fontsize=8,
#                                 verticalalignment='top',
#                                 horizontalalignment='right',
#                                 transform=diff.transAxes)
#
#         fig.tight_layout()
#         #plt.show()
#         plt.savefig(self.data_path + self.config_file + ".pdf")
#
#
# class wavelet_graphs(object):
#     def __init__(self,config_file,data_path):
#         self.config_file = config_file
#         self.data_path   = data_path
#         self.sipm_polar = 0
#
#     def measure(self,event,**kargs):
#
#         roi_size    = kargs['roi_size']   # = 32
#         roi_height  = kargs['roi_height'] # = 16
#         offset      = kargs['offset']     # int(self.sipm_polar[0,0])
#         radius      = kargs['radius']     # 161
#         SiPM_Matrix = kargs['SiPM_Matrix'] # with OFFSET !!!!
#         d_TE        = kargs['d_TE']
#         d_recons    = kargs['d_recons']
#
#
#         max_sipm = np.argwhere(d_TE[event,:] == np.max(d_TE[event,:]))+offset
#         x_y  = np.argwhere(SiPM_Matrix == max_sipm[0])
#
#         # Build ROIs for subevents
#         roi_matrix1  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1],axis=1)[:,0:roi_size].astype(int)
#         roi_matrix2  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1]-SiPM_Matrix.shape[1]//2,axis=1)[:,0:roi_size].astype(int)
#
#         # Check if there is something on the opposite side
#         #if (np.max(d_TE[event,roi_matrix2-offset]) == 0):
#         #    roi_matrix2 = np.array([[]])
#
#         roi_center1 = roi_matrix1[ roi_height//2, roi_size//2 ]
#         roi_center2 = roi_matrix2[ roi_height//2, roi_size//2 ]
#
#         # Get subevent values
#         subevent1_o = d_TE[event,roi_matrix1-offset]
#         subevent1_r = d_recons[event,roi_matrix1-offset]
#         if (roi_matrix2.size > 0):
#             subevent2_o = d_TE[event,roi_matrix2-offset]
#             subevent2_r = d_recons[event,roi_matrix2-offset]
#         else:
#             subevent2_o = np.array([[]])
#             subevent2_r = np.array([[]])
#
#         # Measure parameters
#         # No need to normalize z
#         z_mean1_o =np.mean(subevent1_o*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_o =np.mean(subevent2_o*self.sipm_polar[roi_matrix2-offset,2])
#         z_mean1_r =np.mean(subevent1_r*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_r =np.mean(subevent2_r*self.sipm_polar[roi_matrix2-offset,2])
#
#         # Phi must me normalized to avoid -pi,+pi effects
#         phi_mean1_o = np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_o = np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#         phi_mean1_r = np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_r = np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#
#         z_std1_o   = np.sqrt(np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_o)**2))
#         z_std2_o   = np.sqrt(np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_o)**2))
#         z_std1_r   = np.sqrt(np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_r)**2))
#         z_std2_r   = np.sqrt(np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_r)**2))
#
#         phi_std1_o = np.sqrt(np.mean(subevent1_o*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_o)**2)))
#         phi_std2_o = np.sqrt(np.mean(subevent2_o*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_o)**2)))
#         phi_std1_r = np.sqrt(np.mean(subevent1_r*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_r)**2)))
#         phi_std2_r = np.sqrt(np.mean(subevent2_r*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_r)**2)))
#
#         phi_std1_cart_o = phi_std1_o * radius
#         phi_std2_cart_o = phi_std2_o * radius
#         phi_std1_cart_r = phi_std1_r * radius
#         phi_std2_cart_r = phi_std2_r * radius
#
#         sigma_comb1_o = np.sqrt(z_std1_o**2+phi_std1_cart_o**2)
#         sigma_comb2_o = np.sqrt(z_std2_o**2+phi_std2_cart_o**2)
#         sigma_comb1_r = np.sqrt(z_std1_r**2+phi_std1_cart_r**2)
#         sigma_comb2_r = np.sqrt(z_std2_r**2+phi_std2_cart_r**2)
#
#         out_o, out_r = {},{}
#
#         out_o['z_mean1']     = z_mean1_o        ; out_o['z_mean2']     = z_mean2_o
#         out_o['phi_mean1']   = phi_mean1_o      ; out_o['phi_mean2']   = phi_mean2_o
#         out_o['z_std1']      = z_std1_o         ; out_o['z_std2']      = z_std2_o
#         out_o['phi_std1']    = phi_std1_cart_o  ; out_o['phi_std2']    = phi_std2_cart_o
#         out_o['sigma_comb1'] = sigma_comb1_o    ; out_o['sigma_comb2'] = sigma_comb2_o
#
#         out_r['z_mean1']     = z_mean1_r        ; out_r['z_mean2']     = z_mean2_r
#         out_r['phi_mean1']   = phi_mean1_r      ; out_r['phi_mean2']   = phi_mean2_r
#         out_r['z_std1']      = z_std1_r         ; out_r['z_std2']      = z_std2_r
#         out_r['phi_std1']    = phi_std1_cart_r  ; out_r['phi_std2']    = phi_std2_cart_r
#         out_r['sigma_comb1'] = sigma_comb1_r    ; out_r['sigma_comb2'] = sigma_comb2_r
#
#         return out_o, out_r
#
#
#
#     def __call__(self,roi_size,roi_height):
#
#         CONFIG = CFG.SIM_DATA(filename = self.data_path + \
#                                          self.config_file+".json",
#                               read     = True)
#         orig_array=[]
#         enco_array=[]
#         TE_data_array=[]
#         ENC_data_array=[]
#         W_data_array=[]
#
#         n_files             = CONFIG.data['ENVIRONMENT']['n_files']
#         path                = CONFIG.data['ENVIRONMENT']['path_to_files']
#         MC_out_file_name    = CONFIG.data['ENVIRONMENT']['MC_out_file_name']
#         radius              = CONFIG.data['TOPOLOGY']['radius_ext']
#         n_rows              = CONFIG.data['TOPOLOGY']['n_rows']
#
#
#         # Sensor Positions
#         sipm_cart = np.array(pd.read_hdf(path + MC_out_file_name + "." + \
#                                          str(n_files[0]).zfill(3) + ".h5",
#                                          key='sensors'))
#         self.sipm_polar = np.zeros(sipm_cart.shape)
#         # Columns -> [sensor,r,z,phi]
#         self.sipm_polar[:,0] = sipm_cart[:,0]
#         self.sipm_polar[:,1] = np.sqrt(np.square(sipm_cart[:,1])+np.square(sipm_cart[:,2]))
#         self.sipm_polar[:,2] = sipm_cart[:,3]
#         self.sipm_polar[:,3] = np.arctan2(sipm_cart[:,2],sipm_cart[:,1])
#
#         # Get SiPM Mapping
#         L1, I, SiPM_Matrix, topo = SM.SiPM_Mapping(CONFIG.data,CONFIG.data['L1']['map_style'])
#         offset = int(self.sipm_polar[0,0])
#         SiPM_Matrix = SiPM_Matrix + offset
#
#
#
#         for j in n_files:
#             d_recons  = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_recons'))
#             d_TE      = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_TE'))
#             d_LL = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_LL'))
#             d_LH = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_LH'))
#             d_HL = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_HL'))
#
#
#
#             for i in range(d_TE.shape[0]):
#
#                 original,encoded = self.measure( event       = i,
#                                                 roi_size    = roi_size,
#                                                 roi_height  = roi_height,
#                                                 offset      = offset,
#                                                 radius      = radius,
#                                                 SiPM_Matrix = SiPM_Matrix,
#                                                 d_TE        = d_TE,
#                                                 d_recons    = d_recons)
#
#                 orig_array.append(original)
#                 enco_array.append(encoded)
#
#                 # Data compression statistics
#                 L1_te_bits = 0
#                 for m in L1:
#                     # Find SiPM included in m
#                     L1_SiPM = np.array([],dtype='int').reshape(n_rows,0)
#                     for asic in m:
#                         L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((n_rows,-1),
#                                             order='F')))
#                     # Read data from out_table
#                     data = d_TE[i,L1_SiPM]
#                     if (np.sum(data>CONFIG.data['L1']['TE'])>0):
#                         L1_te_bits = L1_te_bits + DAQ.L1_outframe_nbits(np.sum(data>CONFIG.data['L1']['TE']))
#
#
#                 TE_data_array.append(L1_te_bits)
#
#                 # LL_data = np.sum(d_LL[i,:]>0)
#                 # LH_data = np.sum(np.abs(d_LH[i,:])>0)
#                 # HL_data = np.sum(np.abs(d_HL[i,:])>0)
#                 # W_shared = np.sum((np.abs(d_HL[i,:])>0)+(np.abs(d_LH[i,:])>0)+(d_LL[i,:]>0))
#                 # W_data_array.append(8 + W_shared*(10+2) + LL_data*12 + LH_data*4 + HL_data*4)
#                  #               N_DATA    N_Pixel + WP        N_LL        N_LH          N_HL
#                  # This MUST BE coherent with DAQ_infinity L1_outframe_nbits_WAV
#
#                 n_bits = 0
#                 for m in range(len(CONFIG.data['L1']['L1_mapping_O'])):
#                     data_aux = [0,0]
#                     data_aux.extend(list(d_LL[i,m*160:(m+1)*160]))
#                     data_aux.extend(list(d_LH[i,m*160:(m+1)*160]))
#                     data_aux.extend(list(d_HL[i,m*160:(m+1)*160]))
#                     if (np.sum(d_LL[i,m*160:(m+1)*160] > 0) > 0):
#                         L1_bits = DAQ.L1_outframe_nbits_WAV(np.array(data_aux),CONFIG.data)
#                         n_bits = n_bits + L1_bits
#
#                 W_data_array.append(n_bits)
#
#         # Processing DONE now ERROR computation
#         orig_np = np.array([np.array(list(orig_array[i].values()),dtype=float) for i in range(len(orig_array))])
#         enco_np = np.array([np.array(list(enco_array[i].values()),dtype=float) for i in range(len(enco_array))])
#         #'sigma_comb2'|'sigma_comb1'|'phi_std1'|'phi_std2'|'phi_mean1'|'phi_mean2'|'z_mean1'|'z_mean2'|'z_std2'|'z_std1'
#
#         Error = orig_np-enco_np
#         z_m_err   = np.concatenate((Error[:,6],Error[:,7]))
#         phi_m_err = np.concatenate((Error[:,4],Error[:,5]))*radius
#         phi_s_err = np.concatenate((Error[:,2],Error[:,3]))
#         z_s_err   = np.concatenate((Error[:,8],Error[:,9]))
#
#         z_m_err   = z_m_err[(z_m_err>-1) * (z_m_err<1) * (z_m_err!=0)]
#         phi_m_err = phi_m_err[(phi_m_err>-1) * (phi_m_err<1) * (phi_m_err!=0)]
#         # Filter to avoid most of "close to pi" errors
#         phi_s_err = phi_s_err[(phi_s_err>-2) * (phi_s_err<2) * (phi_s_err!=0)]
#         z_s_err   = z_s_err[(z_s_err>-2) * (z_s_err<2)* (z_s_err!=0)]
#
#
#         # Now the plotting stuff
#         err_fit = fit_library.GND_fit()
#         g_fit   = fit_library.gauss_fit()
#
#         fig = plt.figure(figsize=(10,10))
#         err_fit(z_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(321),
#                         title = "Z mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         err_fit(phi_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(322),
#                         title = "PHI mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         g_fit(z_s_err,'sqrt')
#         g_fit.plot(axis = fig.add_subplot(323),
#                         title = "Z sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = False, fit = True)
#         g_fit(phi_s_err,'sqrt')
#         g_fit.plot(axis = fig.add_subplot(324),
#                         title = "PHI sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = False, fit = True)
#
#
#         data_sent = fig.add_subplot(325)
#         g_fit(TE_data_array,100)
#         x_data = g_fit.bin_centers[g_fit.bin_centers<4000]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='b')
#         g_fit(W_data_array,100)
#         x_data = g_fit.bin_centers[g_fit.bin_centers<4000]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='r')
#
#         data_sent.set_title("DATA sent in TE mode & ENCODER mode")
#         data_sent.set_xlabel("Number of bits")
#         data_sent.set_ylabel("Red - ENCODER (bits)) / Blue - TE (bits)")
#
#         diff_data = np.array(TE_data_array)-np.array(W_data_array)
#         g_fit(diff_data[diff_data<10000],'sqrt')
#         diff = fig.add_subplot(326)
#         diff.set_yscale('log')
#         g_fit.plot(axis = diff,
#                         title = "Difference in Data sent in both modes",
#                         xlabel = "Number of Words",
#                         ylabel = "Hits",
#                         res = False, fit = False)
#
#         diff.text(0.99,0.97,(("DATA SENT in TE MODE  = %d bits\n" + \
#                               "DATA SENT in ENCODER MODE = %d bits\n" + \
#                               "COMPRESS RATIO = %f \n") % \
#                               (np.sum(TE_data_array),np.sum(W_data_array),
#                                float(np.sum(W_data_array))/float(np.sum(TE_data_array)))),
#                                 fontsize=8,
#                                 verticalalignment='top',
#                                 horizontalalignment='right',
#                                 transform=diff.transAxes)
#
#         fig.tight_layout()
#         #plt.show()
#         plt.savefig(self.data_path + self.config_file + ".pdf")
#
#
# class wavelet_graphs_PA(object):
#     def __init__(self,config_file,data_path):
#         self.config_file = config_file
#         self.data_path   = data_path
#         self.sipm_polar = 0
#
#     def measure(self,event,**kargs):
#
#         roi_size    = kargs['roi_size']   # = 32
#         roi_height  = kargs['roi_height'] # = 16
#         offset      = kargs['offset']     # int(self.sipm_polar[0,0])
#         radius      = kargs['radius']     # 161
#         SiPM_Matrix = kargs['SiPM_Matrix'] # with OFFSET !!!!
#         d_TE        = kargs['d_TE']
#         d_recons    = kargs['d_recons']
#
#
#         max_sipm = np.argwhere(d_TE[event,:] == np.max(d_TE[event,:]))+offset
#         x_y  = np.argwhere(SiPM_Matrix == max_sipm[0])
#
#         # Build ROIs for subevents
#         roi_matrix1  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1],axis=1)[:,0:roi_size].astype(int)
#         roi_matrix2  = np.roll(SiPM_Matrix,roi_size//2-x_y[0,1]-SiPM_Matrix.shape[1]//2,axis=1)[:,0:roi_size].astype(int)
#
#         # Check if there is something on the opposite side
#         #if (np.max(d_TE[event,roi_matrix2-offset]) == 0):
#         #    roi_matrix2 = np.array([[]])
#
#         roi_center1 = roi_matrix1[ roi_height//2, roi_size//2 ]
#         roi_center2 = roi_matrix2[ roi_height//2, roi_size//2 ]
#
#         # Get subevent values
#         subevent1_o = d_TE[event,roi_matrix1-offset]
#         subevent1_r = d_recons[event,roi_matrix1-offset]
#         if (roi_matrix2.size > 0):
#             subevent2_o = d_TE[event,roi_matrix2-offset]
#             subevent2_r = d_recons[event,roi_matrix2-offset]
#         else:
#             subevent2_o = np.array([[]])
#             subevent2_r = np.array([[]])
#
#         # Measure parameters
#         # No need to normalize z
#         z_mean1_o =np.mean(subevent1_o*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_o =np.mean(subevent2_o*self.sipm_polar[roi_matrix2-offset,2])
#         z_mean1_r =np.mean(subevent1_r*self.sipm_polar[roi_matrix1-offset,2])
#         z_mean2_r =np.mean(subevent2_r*self.sipm_polar[roi_matrix2-offset,2])
#
#         # Phi must me normalized to avoid -pi,+pi effects
#         phi_mean1_o = np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_o = np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#         phi_mean1_r = np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]))
#         phi_mean2_r = np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]))
#
#         z_std1_o   = np.sqrt(np.mean(subevent1_o*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_o)**2))
#         z_std2_o   = np.sqrt(np.mean(subevent2_o*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_o)**2))
#         z_std1_r   = np.sqrt(np.mean(subevent1_r*(self.sipm_polar[roi_matrix1-offset,2]-z_mean1_r)**2))
#         z_std2_r   = np.sqrt(np.mean(subevent2_r*(self.sipm_polar[roi_matrix2-offset,2]-z_mean2_r)**2))
#
#         phi_std1_o = np.sqrt(np.mean(subevent1_o*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_o)**2)))
#         phi_std2_o = np.sqrt(np.mean(subevent2_o*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_o)**2)))
#         phi_std1_r = np.sqrt(np.mean(subevent1_r*((self.sipm_polar[roi_matrix1-offset,3]-self.sipm_polar[roi_center1-offset,3]-phi_mean1_r)**2)))
#         phi_std2_r = np.sqrt(np.mean(subevent2_r*((self.sipm_polar[roi_matrix2-offset,3]-self.sipm_polar[roi_center2-offset,3]-phi_mean2_r)**2)))
#
#         phi_std1_cart_o = phi_std1_o * radius
#         phi_std2_cart_o = phi_std2_o * radius
#         phi_std1_cart_r = phi_std1_r * radius
#         phi_std2_cart_r = phi_std2_r * radius
#
#         sigma_comb1_o = np.sqrt(z_std1_o**2+phi_std1_cart_o**2)
#         sigma_comb2_o = np.sqrt(z_std2_o**2+phi_std2_cart_o**2)
#         sigma_comb1_r = np.sqrt(z_std1_r**2+phi_std1_cart_r**2)
#         sigma_comb2_r = np.sqrt(z_std2_r**2+phi_std2_cart_r**2)
#
#         out_o, out_r = {},{}
#
#         out_o['z_mean1']     = z_mean1_o        ; out_o['z_mean2']     = z_mean2_o
#         out_o['phi_mean1']   = phi_mean1_o      ; out_o['phi_mean2']   = phi_mean2_o
#         out_o['z_std1']      = z_std1_o         ; out_o['z_std2']      = z_std2_o
#         out_o['phi_std1']    = phi_std1_cart_o  ; out_o['phi_std2']    = phi_std2_cart_o
#         out_o['sigma_comb1'] = sigma_comb1_o    ; out_o['sigma_comb2'] = sigma_comb2_o
#
#         out_r['z_mean1']     = z_mean1_r        ; out_r['z_mean2']     = z_mean2_r
#         out_r['phi_mean1']   = phi_mean1_r      ; out_r['phi_mean2']   = phi_mean2_r
#         out_r['z_std1']      = z_std1_r         ; out_r['z_std2']      = z_std2_r
#         out_r['phi_std1']    = phi_std1_cart_r  ; out_r['phi_std2']    = phi_std2_cart_r
#         out_r['sigma_comb1'] = sigma_comb1_r    ; out_r['sigma_comb2'] = sigma_comb2_r
#
#         return out_o, out_r
#
#     def L1_outframe_nbits(self, data, frame_type=1, n_CH=7,
#                                 TDCmin=26, CH=13, QDC=10, Subthr_sum=10):
#         ########################################################################
#         # DATAFRAME FIELDS
#         # FRAME TYPE  | n_CH | TDCmin | n_CH * [CH | QDC] | Subthr sum(QDC)
#         #     1b      |  7b  |  26b   | n_CH * (13b+10b)  | 10 bits B_QDC
#         ########################################################################
#         if data>0:
#             c=1
#         else:
#             c=0
#         return (frame_type + c*n_CH + TDCmin + data*(CH+QDC) + Subthr_sum)
#
#
#     def L1_outframe_nbits_WAV(self, data, CFG, frame_type=1, n_CH=7,
#                                 TDCmin=26, CH=10, QDC=10, Subthr_sum=10):
#     ########################################################################
#     # DATAFRAME FIELDS
#     # TYPE  | n_CH | TDCmin | n_CH  * [PIX | PW  |  LL | LH | HL]  | Sbth sum(QDC)
#     #  1b   |  7b  |  26b   | n_PIX * (10b + 2b  + 12b + 4b   4b)  | 10 b B_QDC
#     ########################################################################
#
#
#         TW = CFG['L1']['TW']
#         bs = CFG['L1']['wav_blocksize']
#         qw = CFG['L1']['QW']
#         LL_data = np.sum(np.abs(data[2:bs+2])>TW[0])
#         LH_data = np.sum(np.abs(data[bs+2:2*bs+2])>TW[1])
#         HL_data = np.sum(np.abs(data[2*bs+2:3*bs+2])>TW[2])
#         HH_data = np.sum(np.abs(data[3*bs+2:4*bs+2])>TW[3])
#
#         # W_shared = np.sum((np.abs(data[2:bs+2])>TW[0])+\
#         #                   (np.abs(data[bs+2:2*bs+2])>TW[1])+\
#         #                   (np.abs(data[2*bs+2:3*bs+2])>TW[2])+\
#         #                   (np.abs(data[3*bs+2:4*bs+2])>TW[3]))
#
#         W_shared = LL_data + LH_data + HL_data + HH_data
#
#         BITS = frame_type + TDCmin \
#              + n_CH    + W_shared*(CH + 2) + LL_data*qw[0] + LH_data*qw[1] \
#              + HL_data*qw[2] + HH_data*qw[3] + Subthr_sum
#
#
#         return BITS
#
#
#     def __call__(self,roi_size,roi_height):
#
#         CONFIG = CFG.SIM_DATA(filename = self.data_path + \
#                                          self.config_file+".json",
#                               read     = True)
#         orig_array=[]
#         enco_array=[]
#         TE_data_array=[]
#         ENC_data_array=[]
#         W_data_array=[]
#
#         n_files             = CONFIG.data['ENVIRONMENT']['n_files']
#         path                = CONFIG.data['ENVIRONMENT']['path_to_files']
#         MC_out_file_name    = CONFIG.data['ENVIRONMENT']['MC_out_file_name']
#         radius              = CONFIG.data['TOPOLOGY']['radius_ext']
#         n_rows              = CONFIG.data['TOPOLOGY']['n_rows']
#
#
#         # Sensor Positions
#         sipm_cart = np.array(pd.read_hdf(path + MC_out_file_name + "." + \
#                                          str(n_files[0]).zfill(3) + ".h5",
#                                          key='sensors'))
#         self.sipm_polar = np.zeros(sipm_cart.shape)
#         # Columns -> [sensor,r,z,phi]
#         self.sipm_polar[:,0] = sipm_cart[:,0]
#         self.sipm_polar[:,1] = np.sqrt(np.square(sipm_cart[:,1])+np.square(sipm_cart[:,2]))
#         self.sipm_polar[:,2] = sipm_cart[:,3]
#         self.sipm_polar[:,3] = np.arctan2(sipm_cart[:,2],sipm_cart[:,1])
#
#         # Get SiPM Mapping
#         L1, I, SiPM_Matrix, topo = SM.SiPM_Mapping(CONFIG.data,CONFIG.data['L1']['map_style'])
#         offset = int(self.sipm_polar[0,0])
#         SiPM_Matrix = SiPM_Matrix + offset
#
#
#
#         for j in n_files:
#             d_recons  = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_recons'))
#             d_TE      = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_TE'))
#             d_LL = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_LL'))
#             d_LH = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_LH'))
#             d_HL = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_HL'))
#             d_HH = np.array(pd.read_hdf(path + MC_out_file_name + "." + str(j).zfill(3) + ".h5",
#                                  key='MC_encoded_HH'))
#
#
#
#             for i in range(d_TE.shape[0]):
#
#                 original,encoded = self.measure( event       = i,
#                                                 roi_size    = roi_size,
#                                                 roi_height  = roi_height,
#                                                 offset      = offset,
#                                                 radius      = radius,
#                                                 SiPM_Matrix = SiPM_Matrix,
#                                                 d_TE        = d_TE,
#                                                 d_recons    = d_recons)
#
#                 orig_array.append(original)
#                 enco_array.append(encoded)
#
#                 # Data compression statistics
#                 L1_te_bits = 0
#                 for m in L1:
#                     # Find SiPM included in m
#                     L1_SiPM = np.array([],dtype='int').reshape(n_rows,0)
#                     for asic in m:
#                         L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((n_rows,-1),
#                                             order='F')))
#                     # Read data from out_table
#                     data = d_TE[i,L1_SiPM]
#                     if (np.sum(data>CONFIG.data['L1']['TE'])>0):
#                         L1_te_bits = L1_te_bits + self.L1_outframe_nbits(np.sum(data>CONFIG.data['L1']['TE']))
#
#
#                 TE_data_array.append(L1_te_bits)
#
#                 # LL_data = np.sum(d_LL[i,:]>0)
#                 # LH_data = np.sum(np.abs(d_LH[i,:])>0)
#                 # HL_data = np.sum(np.abs(d_HL[i,:])>0)
#                 # W_shared = np.sum((np.abs(d_HL[i,:])>0)+(np.abs(d_LH[i,:])>0)+(d_LL[i,:]>0))
#                 # W_data_array.append(8 + W_shared*(10+2) + LL_data*12 + LH_data*4 + HL_data*4)
#                  #               N_DATA    N_Pixel + WP        N_LL        N_LH          N_HL
#                  # This MUST BE coherent with DAQ_infinity L1_outframe_nbits_WAV
#
#                 n_bits = 0
#                 for m in range(len(CONFIG.data['L1']['L1_mapping_O'])):
#                     data_aux = [0,0]
#                     data_aux.extend(list(d_LL[i,m*160:(m+1)*160]))
#                     data_aux.extend(list(d_LH[i,m*160:(m+1)*160]))
#                     data_aux.extend(list(d_HL[i,m*160:(m+1)*160]))
#                     if (np.sum(d_LL[i,m*160:(m+1)*160] > 0) > 0):
#                         L1_bits = self.L1_outframe_nbits_WAV(np.array(data_aux),CONFIG.data)
#                         n_bits = n_bits + L1_bits
#
#                 W_data_array.append(n_bits)
#
#         # Processing DONE now ERROR computation
#         orig_np = np.array([np.array(list(orig_array[i].values()),dtype=float) for i in range(len(orig_array))])
#         enco_np = np.array([np.array(list(enco_array[i].values()),dtype=float) for i in range(len(enco_array))])
#         #'sigma_comb2'|'sigma_comb1'|'phi_std1'|'phi_std2'|'phi_mean1'|'phi_mean2'|'z_mean1'|'z_mean2'|'z_std2'|'z_std1'
#
#         Error = orig_np-enco_np
#         z_m_err   = np.concatenate((Error[:,6],Error[:,7]))
#         phi_m_err = np.concatenate((Error[:,4],Error[:,5]))*radius
#         phi_s_err = np.concatenate((Error[:,2],Error[:,3]))
#         z_s_err   = np.concatenate((Error[:,8],Error[:,9]))
#
#         z_m_err   = z_m_err[(z_m_err>-1) * (z_m_err<1) * (z_m_err!=0)]
#         phi_m_err = phi_m_err[(phi_m_err>-1) * (phi_m_err<1) * (phi_m_err!=0)]
#         # Filter to avoid most of "close to pi" errors
#         phi_s_err = phi_s_err[(phi_s_err>-2) * (phi_s_err<2) * (phi_s_err!=0)]
#         z_s_err   = z_s_err[(z_s_err>-2) * (z_s_err<2)* (z_s_err!=0)]
#
#
#         # Now the plotting stuff
#         err_fit = fit_library.GND_fit()
#         g_fit   = fit_library.gauss_fit()
#
#         fig = plt.figure(figsize=(10,10))
#         err_fit(z_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(321),
#                         title = "Z mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         err_fit(phi_m_err,'sqrt')
#         err_fit.plot(axis = fig.add_subplot(322),
#                         title = "PHI mean ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = True, fit = True)
#         g_fit(z_s_err,'sqrt')
#         g_fit.plot(axis = fig.add_subplot(323),
#                         title = "Z sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = False, fit = True)
#         g_fit(phi_s_err,'sqrt')
#         g_fit.plot(axis = fig.add_subplot(324),
#                         title = "PHI sigma ERROR",
#                         xlabel = "mm",
#                         ylabel = "Hits",
#                         res = False, fit = True)
#
#
#         data_sent = fig.add_subplot(325)
#         g_fit(TE_data_array,100)
#         x_data = g_fit.bin_centers[g_fit.bin_centers<4000]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='b')
#         g_fit(W_data_array,100)
#         x_data = g_fit.bin_centers[g_fit.bin_centers<4000]
#         data_sent.bar(x_data,g_fit.hist[:len(x_data)],color='r')
#
#         data_sent.set_title("DATA sent in TE mode & ENCODER mode")
#         data_sent.set_xlabel("Number of bits")
#         data_sent.set_ylabel("Red - ENCODER (bits)) / Blue - TE (bits)")
#
#         diff_data = np.array(TE_data_array)-np.array(W_data_array)
#         g_fit(diff_data[diff_data<10000],'sqrt')
#         diff = fig.add_subplot(326)
#         diff.set_yscale('log')
#         g_fit.plot(axis = diff,
#                         title = "Difference in Data sent in both modes",
#                         xlabel = "Number of Words",
#                         ylabel = "Hits",
#                         res = False, fit = False)
#
#         diff.text(0.99,0.97,(("DATA SENT in TE MODE  = %d bits\n" + \
#                               "DATA SENT in ENCODER MODE = %d bits\n" + \
#                               "COMPRESS RATIO = %f \n") % \
#                               (np.sum(TE_data_array),np.sum(W_data_array),
#                                float(np.sum(W_data_array))/float(np.sum(TE_data_array)))),
#                                 fontsize=8,
#                                 verticalalignment='top',
#                                 horizontalalignment='right',
#                                 transform=diff.transAxes)
#
#         fig.tight_layout()
#         #plt.show()
#         plt.savefig(self.data_path + self.config_file + ".pdf")
#
#
#
#
# class infinity_graphs_WP(object):
#     """ Data Analysis and Graphs generation
#     """
#     def __init__(self,config_file,data_path):
#         self.config_file = config_file
#         self.data_path   = data_path
#
#     def __call__(self):
#
#         # Read first config_file to get n_L1 (same for all files)
#         config_file = self.data_path + self.config_file[0] + ".json"
#         CG   = CFG.SIM_DATA(filename = config_file,read = True)
#         CG   = CG.data
#         n_L1 = np.array(CG['L1']['L1_mapping_O']).shape[0]
#
#         logA         = np.array([]).reshape(0,2)
#         logB         = np.array([]).reshape(0,2)
#         logC         = np.array([]).reshape(0,2)
#         log_channels = np.array([]).reshape(0,2)
#         log_outlink  = np.array([]).reshape(0,2)
#         in_time      = np.array([]).reshape(0,1)
#         out_time     = np.array([]).reshape(0,1)
#         lost         = np.array([]).reshape(0,4)
#         compress     = np.array([]).reshape(0,1)
#         frame_frag   = np.array([]).reshape(0,n_L1+1)
#
#         for i in self.config_file:
#             # jsonname = string.replace(i,"/","_")
#             # jsonname = string.replace(jsonname,".","")
#             start = i.rfind("/")
#             jsonname = i[start+1:]
#
#             config_file2 = self.data_path + i + ".json"
#             CG = CFG.SIM_DATA(filename = config_file2,read = True)
#             CG = CG.data
#             chain = CG['ENVIRONMENT']['out_file_name'][CG['ENVIRONMENT']['out_file_name'].rfind("./")+2:]
#             filename = chain + "_" + jsonname + ".h5"
#             filename = self.data_path + filename
#
#             logA         = np.vstack([logA,np.array(pd.read_hdf(filename,key='logA'))])
#             logB         = np.vstack([logB,np.array(pd.read_hdf(filename,key='logB'))])
#             logC         = np.vstack([logC,np.array(pd.read_hdf(filename,key='logC'))])
#             log_channels = np.vstack([log_channels,np.array(pd.read_hdf(filename,key='log_channels'))])
#             log_outlink  = np.vstack([log_outlink,np.array(pd.read_hdf(filename,key='log_outlink'))])
#             in_time      = np.vstack([in_time,np.array(pd.read_hdf(filename,key='in_time'))])
#             out_time     = np.vstack([out_time,np.array(pd.read_hdf(filename,key='out_time'))])
#             lost         = np.vstack([lost,np.array(pd.read_hdf(filename,key='lost'))])
#             compress     = np.vstack([compress,np.array(pd.read_hdf(filename,key='compress'))])
#             frame_frag   = np.vstack([frame_frag,np.array(pd.read_hdf(filename,key='frame_frag'))])
#
#         latency_L1 = logA[:,1]
#         latency    = out_time-in_time
#
#         print ("LOST DATA PRODUCER -> CH      = %d" % (lost[:,0].sum()))
#         print ("LOST DATA CHANNELS -> OUTLINK = %d" % (lost[:,1].sum()))
#         print ("LOST DATA OUTLINK  -> L1      = %d" % (lost[:,2].sum()))
#         print ("LOST DATA L1A -> L1B          = %d" % (lost[:,3].sum()))
#
#         WC_CH_FIFO    = float(max(log_channels[:,0])/CG['TOFPET']['IN_FIFO_depth'])*100
#         WC_OLINK_FIFO = float(max(log_outlink[:,0])/CG['TOFPET']['OUT_FIFO_depth'])*100
#         WC_L1_A_FIFO  = float(max(logA[:,0])/CG['L1']['FIFO_L1a_depth'])*100
#         WC_L1_B_FIFO  = float(max(logB[:,0])/CG['L1']['FIFO_L1b_depth'])*100
#
#
#         print ("\n \n BYE \n \n")
#
#         fit = fit_library.gauss_fit()
#         fig = plt.figure(figsize=(20,10))
#
#         fit(log_channels[:,0],range(1,CG['TOFPET']['IN_FIFO_depth']+2))
#         fit.plot(axis = fig.add_subplot(341),
#                 title = "ASICS Channel Input analog FIFO (4)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(341).set_yscale('log')
#         fig.add_subplot(341).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(341).text(0.99,0.97,(("ASIC Input FIFO reached %.1f %%" % \
#                                                 (WC_CH_FIFO))),
#                                                 fontsize=9,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(341).transAxes)
#
#         fit(log_outlink[:,0],range(1,CG['TOFPET']['OUT_FIFO_depth'],8))
#         fit.plot(axis = fig.add_subplot(342),
#                 title = "ASICS Channels -> Outlink",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(342).set_yscale('log')
#         fig.add_subplot(342).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(342).text(0.99,0.97,(("ASIC Outlink FIFO reached %.1f %%" % \
#                                                 (WC_OLINK_FIFO))),
#                                                 fontsize=9,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(342).transAxes)
#
#         fit(logA[:,0],range(1,CG['L1']['FIFO_L1a_depth']))
#         fit.plot(axis = fig.add_subplot(346),
#                 title = "ASICS -> L1A (FIFOA)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(346).set_yscale('log')
#         fig.add_subplot(346).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(346).text(0.99,0.97,(("L1_A FIFO reached %.1f %%" % \
#                                                 (WC_L1_A_FIFO))),
#                                                 fontsize=9,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(346).transAxes)
#         fig.add_subplot(3,4,6).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         fit(logB[:,0],range(1,CG['L1']['FIFO_L1b_depth'],2))
#         fit.plot(axis = fig.add_subplot(345),
#                 title = "L1 OUTPUT (FIFOB)",
#                 xlabel = "FIFO Occupancy",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(345).set_yscale('log')
#         fig.add_subplot(345).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(345).text(0.99,0.97,(("L1_B FIFO reached %.1f %%" % \
#                                                 (WC_L1_B_FIFO))),
#                                                 fontsize=9,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(345).transAxes)
#         fig.add_subplot(3,4,5).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         fit(logC[:,0],range(8,int(np.max(logC[:,0]))+4))
#         fit.plot(axis = fig.add_subplot(3,4,10),
#                 title = "Number of Frames per Buffer",
#                 xlabel = "Number of Frames",
#                 ylabel = "Hits",
#                 res = False, fit = True)
#         fig.add_subplot(3,4,10).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         fit(latency/1000.0,40)
#         fit.plot(axis = fig.add_subplot(343),
#                 title = "Total Data Latency",
#                 xlabel = "Latency in microseconds",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(343).text(0.99,0.8,(("WORST = %1.2f $\mu$s" % \
#                                                 (max(latency)/1000.0))),
#                                                 fontsize=8,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(343).transAxes)
#         fig.add_subplot(343).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         new_axis = fig.add_subplot(347)
#         x_data = fit.bin_centers
#         y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
#         new_axis.plot(x_data,y_data)
#         new_axis.set_ylim((0.9,1.0))
#         new_axis.set_xlabel("Latency in nanoseconds")
#         new_axis.set_ylabel("Percentage of Recovered Data")
#         new_axis.text(0.05,0.9,(("LOST DATA PRODUCER -> CH           = %d\n" + \
#                                  "LOST DATA CHANNELS -> OUTLINK  = %d\n" + \
#                                  "LOST DATA OUTLINK -> L1                = %d\n" + \
#                                  "LOST DATA L1A -> L1B                      = %d\n") % \
#                                 (lost[:,0].sum(),
#                                  lost[:,1].sum(),
#                                  lost[:,2].sum().sum(),
#                                  lost[:,3].sum().sum())
#                                 ),
#                                 fontsize=9,
#                                 verticalalignment='top',
#                                 horizontalalignment='left',
#                                 transform=new_axis.transAxes)
#
#         fit(latency_L1/1000.0,30)
#         fit.plot(axis = fig.add_subplot(349),
#                 title = "L1 input Data Latency",
#                 xlabel = "Latency in microseconds",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(349).text(0.99,0.8,(("WORST = %1.2f $\mu$s" % \
#                                                 (max(latency_L1)/1000.0))),
#                                                 fontsize=9,
#                                                 verticalalignment='top',
#                                                 horizontalalignment='right',
#                                                 transform=fig.add_subplot(349).transAxes)
#         fig.add_subplot(349).xaxis.set_major_locator(MaxNLocator(integer=True))
#         fig.add_subplot(349).set_yscale('log')
#
#         fit(compress,30) #range(int(np.max(compress))))
#         fit.plot(axis = fig.add_subplot(344),
#                 title = "Data Frame Length",
#                 xlabel = "Bits",
#                 ylabel = "Hits",
#                 res = False,
#                 fit = False)
#         fig.add_subplot(344).set_yscale('log')
#         fig.add_subplot(344).xaxis.set_major_locator(MaxNLocator(integer=True))
#         # TOTAL NUMBER OF BITS vs COMPRESS EFFICIENCY
#         # A = np.arange(0,np.max(compress))
#         # D_data = [DAQ.L1_outframe_nbits(i) for i in A]
#         # #D_data = 1 + 7*(A>0) + A * 23 + 10     #see DAQ_infinity
#         # D_save = (A-1)*10
#         #
#         #
#         # B_data = np.multiply(D_data,fit.hist)
#         # B_save = np.multiply(D_save,fit.hist)
#         # B_save[0]=0
#         # B_save[1]=0
#         # new_axis_2 = fig.add_subplot(348)
#         # x_data = fit.bin_centers
#         # new_axis_2.bar(x_data,B_data,color='r')
#         # new_axis_2.bar(x_data,B_save,color='b')
#         # new_axis_2.set_title("Data sent vs frame length")
#         # new_axis_2.set_xlabel("Length of frame in QDC data")
#         # new_axis_2.set_ylabel("Red - Data sent (bits) / Blue - Data saved (bits)")
#         #
#         # new_axis_2.text(0.99,0.97,(("TOTAL DATA SENT = %d bits\n" + \
#         #                          "DATA REDUCTION  = %d bits\n" + \
#         #                          "COMPRESS RATIO = %f \n") % \
#         #                         (np.sum(B_data),np.sum(B_save),float(np.sum(B_data))/float(np.sum(B_save)+np.sum(B_data)))),
#         #                         fontsize=8,
#         #                         verticalalignment='top',
#         #                         horizontalalignment='right',
#         #                         transform=new_axis_2.transAxes)
#
#
#         ############### FRAME FRAGMENTATION ANALYSIS ##########################
#         # TIME FRAGMENTATION
#         frag_matrix = frame_frag[1:,1:]
#         frag_matrix = frag_matrix.reshape(-1)
#         frag_matrix = (frag_matrix>0)*frag_matrix
#         fit(frag_matrix,range(1,int(np.max(frag_matrix))+2))
#         print int(np.max(frag_matrix))
#         fit.plot(axis = fig.add_subplot(3,4,11),
#                 title = "Frame Fragmentation - (TIME)",
#                 xlabel = "Frame pieces (Buffers)",
#                 ylabel = "Hits",
#                 res = False, fit = False)
#         fig.add_subplot(3,4,11).set_yscale('log')
#         fig.add_subplot(3,4,11).set_ylim(bottom=1)
#         fig.add_subplot(3,4,11).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         # SPATIAL FRAGMENTATION
#         cluster_record = []
#         frag_matrix = frame_frag[:,1:]
#
#         for ev in frag_matrix:
#             clusters_aux = np.zeros((1,2))
#             clusters = np.array([]).reshape(0,2)
#             flag = False
#             L1_count = 0
#             # First element , Last element
#             for L1 in ev:
#                 if (L1 > 0):
#                     # Cluster detected
#                     if (flag == False):
#                         # First element in the cluster
#                         clusters_aux[0,0] = L1_count
#                         clusters_aux[0,1] = L1_count
#                         flag = True
#                     else:
#                         # Inside the cluster
#                         clusters_aux[0,1] = L1_count
#                         flag = True
#
#                     if (L1_count == (n_L1-1)):
#                         clusters = np.vstack([clusters,clusters_aux])
#                 else:
#                     if (flag == True):
#                         # End of cluster
#                         flag = False
#                         clusters = np.vstack([clusters,clusters_aux])
#                         clusters_aux = np.zeros((1,2))
#                     else:
#                         # No cluster detected
#                         flag = False
#
#                 L1_count += 1
#
#             # Must solve special case of boundary cluster in circular buffer
#             if ((clusters[0,0] == 0) and (clusters[-1,1] == (L1_count-1))):
#                 clusters[0,0] = clusters[-1,0]
#                 clusters = clusters[:-1,:]
#
#             cluster_record.append(clusters)
#
#         cluster_lengths = []
#         # LET'S FIND CLUSTER LENGTHS
#         for i in cluster_record:
#             for j in i:
#                 if (j[1] >= j[0]):
#                     cluster_lengths.append(int(j[1]-j[0]+1))
#                 else:
#                     cluster_lengths.append(int(j[1]-j[0]+n_L1+1))
#
#         fit(cluster_lengths,range(1,int(np.max(cluster_lengths))+2))
#         fit.plot(axis = fig.add_subplot(3,4,12),
#                 title = "Frame Fragmentation - (SPACE)",
#                 xlabel = "Number of L1 per event",
#                 ylabel = "Hits",
#                 res = False,
#                 fit = False)
#         fig.add_subplot(3,4,12).set_yscale('log')
#         fig.add_subplot(3,4,12).xaxis.set_major_locator(MaxNLocator(integer=True))
#
#
#         fig.tight_layout()
#
#         #plt.savefig(CG['ENVIRONMENT']['out_file_name']+"_"+ filename + ".pdf")
#         plt.savefig(filename + ".pdf")
#
#
#
#
#
# def main():
#     # A = infinity_graphs(["OF_4mm_BUF640_V3"],
#     #                      "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/4mm_pitch/")
#     # A()
#     # start = time.time()
#     #
#     # files = [0,1,2,3,4,5,6,8]
#     #
#     # TEST_c = hdf_compose(  "/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
#     #                        "p_FRSET_", files, 1536)
#     # a,b,c = TEST_c.compose()
#     #
#     # time_elapsed = time.time() - start
#     #
#     # print ("It took %d seconds to compose %d files" % (time_elapsed,
#     #                                                    len(files)))
#
#     #A = wavelet_graphs("./WAVELET_P1/test","/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/5mm_pitch/")
#     #A(roi_size=32,roi_height=16)
#
#     A = infinity_graphs_WP(["./WAVELET_P1/event_sim"],"/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/5mm_pitch/")
#     A()
#     #A(roi_size=32,roi_height=16)
#
#     # A = ENCODER_MAT2HF(path = "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/5mm_pitch/",
#     #                       in_file = "compresores_pitch5mm_rad161mm_1_medio_ver1_export.mat",
#     #                       out_file = "Rafa_2UP.h5",
#     #                       json_file = "Encoder_Test")
#     # A.read()
#     # A.write()

if __name__ == "__main__":
    main()
