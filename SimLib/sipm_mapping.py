
import numpy as np
import scipy as sp
import math

def SiPM_Mapping(param, style):

    # Work out number of SiPMs based on geometry data
    n_sipms_I = param['TOPOLOGY']['sipm_int_row']*param['TOPOLOGY']['n_rows']
    n_sipms_O = param['TOPOLOGY']['sipm_ext_row']*param['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_I + n_sipms_O
    # Number of ASICs calculation: Inner Face + Outer Face // full + partial
    n_asics_I = int(math.ceil(float(n_sipms_I) / float(param['TOFPET']['n_channels'])))
    n_asics_f_I  = n_sipms_I // param['TOFPET']['n_channels']  # Fully used
    n_asics_p_I = n_asics_I - n_asics_f_I                       # Partially used
    n_asics_O = int(math.ceil(float(n_sipms_O) / float(param['TOFPET']['n_channels'])))
    n_asics_f_O  = n_sipms_O // param['TOFPET']['n_channels']
    n_asics_p_O   = n_asics_O - n_asics_f_O      # Number of not fully used ASICs (0 or 1)
    n_asics = n_asics_I + n_asics_O
    # L1 are required with max number of ASICs in param.P['L1']['n_asics']
    # // full + part
    n_L1 = int(math.ceil(float(n_asics) / float(param['L1']['n_asics'])))
    n_L1_f = n_asics // param['L1']['n_asics']
    n_L1_p = n_L1 - n_L1_f

    n_L1_I = int(math.ceil(float(n_asics_I) / float(param['L1']['n_asics'])))
    n_L1_I_f = n_asics_I // param['L1']['n_asics']
    n_L1_I_p = n_L1_I - n_L1_I_f

    n_L1_O = int(math.ceil(float(n_asics_O) / float(param['L1']['n_asics'])))
    n_L1_O_f = n_asics_O // param['L1']['n_asics']
    n_L1_O_p = n_L1_O - n_L1_O_f


    print ("Number of SiPM : %d \nNumber of ASICS : %d " % (n_sipms,n_asics))
    print ("Minimum Number of L1 : %d " % (n_L1))

    SiPM_Matrix_I = np.reshape(np.arange(0,n_sipms_I),
                                (param['TOPOLOGY']['n_rows'],
                                param['TOPOLOGY']['sipm_int_row']))
    SiPM_Matrix_O = np.reshape(np.arange(n_sipms_I,n_sipms),
                                (param['TOPOLOGY']['n_rows'],
                                param['TOPOLOGY']['sipm_ext_row']))
    # SiPM matrixs Inner face and Outer face



    # if style == "striped":
    #     L1_Slice=[]
    #     count = 0
    #     SiPM_ASIC_Slice=[]
    #     # Generate Slice of ASICs (SiPM) for L1
    #     for i in range(topology['n_asics_I']):
    #         if (count < param.P['L1']['n_asics']-1):
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
    #             count += 1
    #         else:
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
    #             L1_Slice.append(SiPM_ASIC_Slice)
    #             SiPM_ASIC_Slice=[]
    #             count = 0
    #     for i in range(topology['n_asics_O']):
    #         if (count < param.P['L1']['n_asics']-1):
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
    #             count += 1
    #         else:
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
    #             L1_Slice.append(SiPM_ASIC_Slice)
    #             SiPM_ASIC_Slice=[]
    #             count = 0
    #
    #     if (topology['n_L1_p'] == 1):
    #         L1_Slice.append(SiPM_ASIC_Slice)


    if style == "mixed":
        L1_Slice=[]
        SiPM_ASIC_Slice=[]
        count = 0

        for i in range(topology['n_asics_I']):
            if (count < param['L1']['n_asics']-2):
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                count += 2
            else:
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                L1_Slice.append(SiPM_ASIC_Slice)
                SiPM_ASIC_Slice=[]
                count = 0

        for i in range(topology['n_asics_O']-topology['n_asics_I']):
            if (count < param['L1']['n_asics']-1):
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                count += 1
            else:
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                L1_Slice.append(SiPM_ASIC_Slice)
                SiPM_ASIC_Slice=[]
                count = 0

        if (topology['n_L1_p'] == 1):
            L1_Slice.append(SiPM_ASIC_Slice)



    if style == "striped":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice=[]
        SiPM_Slice=[]
        extra = 0

        # Generate Slice of ASICs (SiPM) for L1
        for k in range(param['TOPOLOGY']['sipm_int_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_I[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)


        # L1 ASIGNMENT
        L1_I_nasics = n_asics_I // n_L1_I
        L1_O_nasics = n_asics_O // n_L1_O

        extra = n_asics_I - L1_I_nasics * n_L1_I
        if extra > 0:
            inc = 1
        else:
            inc = 0

        for i in range(n_asics_I):
            L1_aux_Slice.append(ASIC_Slice[i])
            count_asic += 1
            if (count_asic == L1_I_nasics + inc):
                L1_Slice.append(L1_aux_Slice)
                L1_aux_Slice = []
                count_asic = 0
                if extra > 1 :
                    extra = extra - 1
                else:
                    inc = 0
        if count_asic > 0:
            L1_Slice.append(L1_aux_Slice)



        count_asic = 0
        L1_aux_Slice = []
        extra = n_asics_O - L1_O_nasics * n_L1_O
        if extra > 0:
            inc = 1
        else:
            inc = 0

        for i in range(n_asics_I,n_asics):
            L1_aux_Slice.append(ASIC_Slice[i])
            count_asic += 1
            if (count_asic == (L1_O_nasics + inc)):
                L1_Slice.append(L1_aux_Slice)
                L1_aux_Slice = []
                count_asic = 0
                if extra > 1 :
                    extra = extra - 1
                else:
                    inc = 0
        if count_asic > 0:
            L1_Slice.append(L1_aux_Slice)




    if style == "striped_2":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice=[]
        SiPM_Slice=[]
        extra = 0

        # Generate Slice of ASICs (SiPM) for L1
        for k in range(param['TOPOLOGY']['sipm_int_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_I[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)


        L1_I = param['L1']['L1_mapping_I']
        L1_O = param['L1']['L1_mapping_O']


        asic_count = 0
        for i in L1_I:
            for j in range(i):
                L1_aux_Slice.append(ASIC_Slice[asic_count])
                asic_count += 1
            L1_Slice.append(L1_aux_Slice)
            L1_aux_Slice=[]

        for i in L1_O:
            for j in range(i):
                L1_aux_Slice.append(ASIC_Slice[asic_count])
                asic_count += 1
            L1_Slice.append(L1_aux_Slice)
            L1_aux_Slice=[]

    if style == "striped_3":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice=[]
        SiPM_Slice=[]
        extra = 0

        # Generate Slice of ASICs (SiPM) for L1
        # for k in range(param['TOPOLOGY']['sipm_int_row']):
        #     for j in range(param['TOPOLOGY']['n_rows']):
        #         SiPM_Slice.append(SiPM_Matrix_I[j,k])
        #         count_ch += 1
        #         if count_ch == param['TOFPET']['n_channels']:
        #             ASIC_Slice.append(SiPM_Slice)
        #             SiPM_Slice = []
        #             count_ch = 0
        # if (count_ch > 0):
        #     ASIC_Slice.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)


        L1_I = param['L1']['L1_mapping_I']
        L1_O = param['L1']['L1_mapping_O']


        asic_count = 0
        for i in L1_I:
            for j in range(i):
                L1_aux_Slice.append(ASIC_Slice[asic_count])
                asic_count += 1
            L1_Slice.append(L1_aux_Slice)
            L1_aux_Slice=[]

        for i in L1_O:
            for j in range(i):
                L1_aux_Slice.append(ASIC_Slice[asic_count])
                asic_count += 1
            L1_Slice.append(L1_aux_Slice)
            L1_aux_Slice=[]


    if style == "mixed_2":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice_I=[]
        ASIC_Slice=[]
        ASIC_Slice_O=[]
        SiPM_Slice=[]
        extra = 0

        # Generate Slice of ASICs (SiPM) for L1
        for k in range(param['TOPOLOGY']['sipm_int_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_I[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice_I.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice_I.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice_O.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice_O.append(SiPM_Slice)


        L1_I =  param['L1']['L1_mapping_I']
        L1_O =  param['L1']['L1_mapping_O']


        asic_count_I = 0
        asic_count_O = 0
        for i in range(len(L1_I)):
            for j in range(L1_I[i]):
                L1_aux_Slice.append(ASIC_Slice_I[asic_count_I])
                asic_count_I += 1
            for j in range(L1_O[i]):
                L1_aux_Slice.append(ASIC_Slice_O[asic_count_O])
                asic_count_O += 1
            L1_Slice.append(L1_aux_Slice)
            L1_aux_Slice=[]

        ASIC_Slice.append(ASIC_Slice_I)
        ASIC_Slice.append(ASIC_Slice_O)



    # Number of ASICs
    print ("Available ASICS = %d" % (len(ASIC_Slice)))
    print ("Connected ASICS = %d" % (np.sum(np.array(L1_I))+np.sum(np.array(L1_O))))
    print ("Instanciated L1 = %d" % (len(L1_Slice)))
    for i in range(len(L1_Slice)):
        print ("L1 number %d has %d ASICs" % (i,len(L1_Slice[i])))


    topology = {'n_sipms_I':n_sipms_I, 'n_sipms_O':n_sipms_O, 'n_sipms': n_sipms,
            'n_asics_I':n_asics_I, 'n_asics_f_I':n_asics_f_I,'n_asics_p_I':n_asics_p_I,
            'n_asics_O':n_asics_O, 'n_asics_f_O':n_asics_f_O,'n_asics_p_O':n_asics_p_O,
            'n_asics':n_asics, 'n_L1':len(L1_Slice)} #, 'n_L1_f':n_L1_f, 'n_L1_p':n_L1_p}


    return L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology
