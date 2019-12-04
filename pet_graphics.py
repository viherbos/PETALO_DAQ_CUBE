import json
import os
import numpy as np
import sys
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui
sys.path.append("../PETALO_DAQ_infinity/")
from SimLib import config_sim as conf
#from SimLib import DAQ_infinity as DAQ
from SimLib import sipm_mapping as DAQ
import argparse

# python pet_graphics.py -f -a -b -j -p p_RING_7mm_6x6_000 MC MC /home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/7mm_pitch/CUBE


class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None, size=7, color=[50,50,50]):
        GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z
        self.size = size
        self.color = color

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = ZrenderText
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtGui.QColor(*self.color))
        self.myFont = QtGui.QFont("Arial", self.size);

        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text, self.myFont)

class DET_SHOW(object):
    def __init__(self,data,app,widget):
        self.sipm = data['SIPM']['size']
        self.app  = app
        self.w    = widget
        self.w.setBackgroundColor([0,0,0])
        self.data = data

    def np2cart(self,p):
        return {'x':p[0],'y':p[1],'z':p[2]}
    def cart2np(self,p):
        return [p['x'],p['y'],p['z']]
    def cyl2np(self,p):
        return [p['r'],p['z'],p['phi']]
    def np2cyl(self,p):
        return {'r':p[0],'z':p[1],'phi':p[2]}

    def cart2cyl(self,p):
        #x=p[0];y=p[1];z=p[2]
        q={}
        q['r'] = np.sqrt(np.square(p['x'])+np.square(p['y']))
        q['z'] = p['z']
        q['phi'] = np.arctan2(p['y'],p['x'])
        return q

    def cyl2cart(self,p):
        q={}
        q['z'] = p['z']
        q['x'] = p['r']*np.cos(p['phi'])
        q['y'] = p['r']*np.sin(p['phi'])
        return q

    def rot_cyl(self,p_array,phi):
        r = np.zeros(p_array.shape)
        for i in range(p_array.shape[0]):
            r[i,:] = self.cyl2np(self.cart2cyl(self.np2cart(p_array[i,:])))
            r[i,2] += phi
            r[i,:] = self.cart2np(self.cyl2cart(self.np2cyl(r[i,:])))
        return r


    def SiPM_QT(self,position,phi,name,photons,max_photons,id=False,
                show_photons=True, MU_LIN=True, TH=0, color2=[0,0,0]):
        s = np.array(self.sipm,dtype=float)
        p = np.array([[s[0]/2,s[1]/2,s[2]/2],
                      [s[0]/2,s[1]/2,-s[2]/2],
                      [-s[0]/2,s[1]/2,s[2]/2],
                      [-s[0]/2,s[1]/2,-s[2]/2],
                      [s[0]/2,-s[1]/2,s[2]/2],
                      [s[0]/2,-s[1]/2,-s[2]/2],
                      [-s[0]/2,-s[1]/2,s[2]/2],
                      [-s[0]/2,-s[1]/2,-s[2]/2]])

        p=self.rot_cyl(p,phi)
        position3d = np.array([position for i in range(p.shape[0])])
        p = p + position3d

        #plt = gl.GLScatterPlotItem(pos=p, color=pg.glColor('w'),size=1)
        MU=25
        if (max_photons == 0):
            color = 0
        else:
            if MU_LIN==True:
                color = int(np.log(1+MU*photons/max_photons)/np.log(1+MU)*200.0)
            else:
                color = int(photons/max_photons*200)

        rgb=np.array([0.75*color+55,color+55,0.25*color+55])
        color  = pg.glColor(*rgb)
        color2 = pg.glColor(*(np.array(color2)))

        meshdata = gl.MeshData(vertexes=p,
                               faces=np.array([ [0,1,4], [5,4,1],
                                                [2,0,6], [4,6,0],
                                                [0,2,3], [1,0,3],
                                                [6,2,3], [7,3,6],
                                                [7,3,5], [1,3,5]]),
                          faceColors=np.array([ color,     color,
                                                color2,    color2,
                                                color2,    color2,
                                                color,     color,
                                                color2,    color2]))

        plt = gl.GLMeshItem(meshdata=meshdata, color = None,
                            edgeColor=pg.glColor('w'),
                            drawEdges=False,
                            smooth=False,
                            drawFaces=True,
                            #shader= 'shaded',#'balloon',#None, #'shaded',
                            glOptions='opaque',
                            computeNormals='False')

        #plt.setGLOptions('additive')
        self.w.addItem(plt)

        if (id==True): #and (int(name)==3512 or int(name)==1000 or int(name)==3511)):
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(name)),size=8,color=np.array([255,0,0]))

            t.setGLViewWidget(self.w)
            self.w.addItem(t)


        text_color = (np.array([rgb[0],255,255])-rgb)*0.75
        text_size = self.w.opts['distance']

        if (show_photons==True and photons>TH) :
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(photons)),size=8,
                            color=text_color)

            t.setGLViewWidget(self.w)
            self.w.addItem(t)


    def __call__(self,sensors,data,event,ident=False,show_photons=True,
                MU_LIN=True,TH=0):
        items = []

        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = DAQ.SiPM_Mapping(self.data,
                                                                            self.data['L1']['map_style'])

        for i in list(self.w.items):
            self.w.removeItem(i)

        max_light = float(data[event,:].max())
        print(max_light)
        count=np.zeros(3)+np.array([20,50,50])
        cnt=0

        # for j in SiPM_Matrix_I: #sensors:
        #     for k in j:
        #         k = k+1000 # Paola's style
        #         i = sensors[int(np.argwhere(sensors[:,0]==k))]
        #         self.SiPM_QT(i[1:].transpose(),
        #                      np.arctan2(i[2],i[1]),i[0],
        #                      data[event,int(i[0]-1000)],
        #                      max_light,
        #                      id = ident,
        #                      show_photons=show_photons,
        #                      MU_LIN=MU_LIN,
        #                      TH=TH
        #                      )

        color_map = [[1,0,0],[0,1,0],[0,0,1],
                     [1,1,0],[0,1,1],[1,0,1],
                     [2.5,0,0],[0,2.5,0],[0,0,2.5],
                     [2.5,2.5,0],[0,2.5,2.5],[2.5,0,2.5]]
        color_i = 100


        for m in L1_Slice:
            # count = [0,0,0]
            # cnt = np.random.randint(0,2)
            # count[cnt] = np.random.randint(80,255)

            for j in m: #ASICs
                for k in j: #SiPMS
                    k = k+1000 # Paola's style
                    i = sensors[int(np.argwhere(sensors[:,0]==k))]

                    if (data[event,int(i[0]-1000)]>1):
                        data_show = data[event,int(i[0]-1000)]
                    else:
                        data_show = 0

                    self.SiPM_QT(i[1:].transpose(),
                                 np.arctan2(i[2],i[1]),i[0],
                                 data_show,
                                 max_light,
                                 id = ident,
                                 show_photons=show_photons,
                                 MU_LIN=MU_LIN,
                                 TH=TH,
                                 color2=[color_i*color_map[cnt][2],
                                         color_i*color_map[cnt][1],
                                         color_i*color_map[cnt][0]]
                                 #color2=[color_i,color_i,color_i]
                                 )
            if cnt < 11:
                cnt += 1
            else:
                cnt = 0



        # t = GLTextItem( X=0, Y=0, Z=0, text=str(event), size=12)
        # t.setGLViewWidget(self.w)
        # self.w.addItem(t)


class graphs_update(object):
    def __init__(self,SIM_CONT,Qtapp,widget,widget2,text,data_TE,data_recons,
                 positions,g_opt):
        self.event = text.value()
        self.Qtapp = Qtapp
        self.widget = widget
        self.widget2 = widget2
        self.positions = positions
        self.data_TE = data_TE
        self.data_recons = data_recons
        self.B = DET_SHOW(SIM_CONT.data,self.Qtapp,self.widget)
        self.B2 = DET_SHOW(SIM_CONT.data,self.Qtapp,self.widget2)
        self.g_opt = g_opt

    def response(self):
        self.B( self.positions, self.data_TE, event=self.event,
           ident=self.g_opt['sipm_id'],
           show_photons=self.g_opt['photons_id'],
           MU_LIN=True,
           TH=0
         )
        self.B2( self.positions, self.data_recons, event=self.event,
           ident=self.g_opt['sipm_id'],
           show_photons=self.g_opt['photons_id'],
           MU_LIN=True,
           TH=0
         )
        self.widget.opts['distance']=550
        self.widget2.opts['distance']=550

        t_recons = GLTextItem( X=20,Y=0,Z=0,text="RECONS "+str(self.event),size=8)
        t_TE = GLTextItem( X=5,Y=0,Z=0,text="TE "+str(self.event),size=8)

        t_recons.setGLViewWidget(self.widget2)
        self.widget2.addItem(t_recons)
        t_TE.setGLViewWidget(self.widget)
        self.widget.addItem(t_TE)

        self.widget.update()
        self.widget2.update()


    def event_update(self):
        self.event = text.value()



def data_graph(SIM_CONT,data_TE,data_recons,positions,g_opt):
    Qtapp  = pg.QtGui.QApplication([])
    window = QtGui.QWidget()

    text = QtGui.QSpinBox()
    text.setRange(0,99999)
    btn = QtGui.QPushButton('SHOW EVENT')
    widget = gl.GLViewWidget()
    widget2 = gl.GLViewWidget()
    graph = graphs_update(SIM_CONT,Qtapp,widget,widget2,text,
                                 data_TE,data_recons,positions,g_opt)

    layout = QtGui.QGridLayout()
    window.setLayout(layout)
    layout.addWidget(text,   0, 0, 1, 1)
    layout.addWidget(btn,    0, 1, 1, 3)
    layout.addWidget(widget, 1, 0, 1, 4)
    layout.addWidget(widget2,2, 0, 1, 4)

    window.resize(768,1024)
    graph.response()


    # Signals connection
    btn.clicked.connect(graph.response)
    text.valueChanged.connect(graph.event_update)

    window.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



if __name__ == '__main__':

    g_opt = {}
    # Argument parser for config file name
    parser = argparse.ArgumentParser(description='PETALO GRAPHICS.')
    parser.add_argument("-f", "--data_file", action="store_true",
                        help="Show events with configuration stored in json file")
    parser.add_argument('arg1', metavar='data_file', nargs='?', help='')

    parser.add_argument("-a", "--window_A", action="store_true",
                        help="WINDOW A")
    parser.add_argument('arg2', metavar='window A', nargs='?', help='')

    parser.add_argument("-b", "--window_B", action="store_true",
                        help="WINDOW B")
    parser.add_argument('arg3', metavar='window B', nargs='?', help='')

    parser.add_argument("-j", "--json_file", action="store_true",
                        help="Json File")
    parser.add_argument('arg4', metavar='Json file', nargs='?', help='')

    parser.add_argument("-p", "--show_photons",action="store_true",
                        help="Show Number of Photons")

    parser.add_argument("-s", "--show_sipmID",action="store_true",
                        help="Show Number of Photons")


    args = parser.parse_args()

    if args.data_file:
        file_name = ''.join(args.arg1)
    else:
        file_name = ""

    if args.window_A:
        window_A = ''.join(args.arg2)
    else:
        window_A = "MC_TE"

    if args.window_B:
        window_B = ''.join(args.arg3)
    else:
        window_B = "MC_recons"

    if args.json_file:
        json_file = ''.join(args.arg4)
    else:
        json_file = "test"

    if args.show_photons:
        g_opt['photons_id'] = True
    else:
        g_opt['photons_id'] = False

    if args.show_sipmID:
        g_opt['sipm_id'] = True
    else:
        g_opt['sipm_id'] = False



    config_file = json_file + ".json"

    SIM_CONT=conf.SIM_DATA(filename=config_file ,read=True)

    path     = SIM_CONT.data['ENVIRONMENT']['path_to_files']

    #filename = SIM_CONT.data['ENVIRONMENT']['MC_out_file_name']+'.'+str(file_n).zfill(3)
    #filename = "./VER5/DAQ_OF5mm_test_REAL"

    positions = np.array(pd.read_hdf(path+file_name+".h5",key='sensor_positions'))
    data_TE = np.array(pd.read_hdf(path+file_name+".h5",key=window_A), dtype = 'int32')
    data_recons = np.array(pd.read_hdf(path+file_name+".h5",key=window_B), dtype = 'int32')
    #data_TE = np.array(pd.read_hdf(path+filename+".h5",key='MC'), dtype = 'int32')
    #data_recons = np.array(pd.read_hdf(path+filename+".h5",key='MC'), dtype = 'int32')


    Qtapp  = pg.QtGui.QApplication([])
    window = QtGui.QWidget()

    text = QtGui.QSpinBox()
    text.setRange(0,99999)
    btn = QtGui.QPushButton('SHOW EVENT')
    widget = gl.GLViewWidget()
    widget2 = gl.GLViewWidget()
    graph = graphs_update(SIM_CONT,Qtapp,widget,widget2,text,
                                 data_TE,data_recons,positions,g_opt)

    layout = QtGui.QGridLayout()
    window.setLayout(layout)
    layout.addWidget(text,   0, 0, 1, 1)
    layout.addWidget(btn,    0, 1, 1, 3)
    layout.addWidget(widget, 1, 0, 1, 4)
    layout.addWidget(widget2,2, 0, 1, 4)

    window.resize(768,1024)
    graph.response()


    # Signals connection
    btn.clicked.connect(graph.response)
    text.valueChanged.connect(graph.event_update)

    window.show()



    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
