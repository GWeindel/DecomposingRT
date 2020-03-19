import scipy.stats as spss
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob,os
import pandas as pd
import ipyparallel
import hddm
from kabuki.analyze import gelman_rubin
import sys
import time
from IPython.display import clear_output

print(os.getcwd())

def wait_watching_stdout(ar, dt=100):
    ## ar: vmap output of the models being run
    ## dt: number of seconds between checking output
    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        print '-' * 30
        print "%.3fs elapsed" % ar.elapsed
        print ""
        for out in ar.stdout: print(out);
        sys.stdout.flush()
        time.sleep(dt)

def run_model(id):
    from patsy import dmatrix
    from pandas import Series
    import numpy as np
    import hddm
    dataHDDM = hddm.load_csv('DDM/dataHDDM.csv')
    dataHDDM = dataHDDM[dataHDDM.exp == "two"]
    dataHDDM["rt"] = np.abs(dataHDDM.rt) #I wrongly flipped errors in data prep file
#    dataHDDM["SAT"] = dataHDDM.apply(lambda row: -0.5 if row['SAT'] == "Accuracy" else 0.5, axis=1)
    dataHDDM["givenResp"] = dataHDDM["response"] #.apply(lambda row: "Left" if row['response'] == 1 else "Right", axis=1)
#    dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1)

    if id < 4:
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1)
        ############## M1
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : 'SAT',
                'a' : 'SAT'}
        inc = ['sv','sz','st']
        model_name = "M1"
        m = hddm.HDDM(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"])
    elif id > 3 and id < 8:
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1)
        ############## M2
        deps = {'sz' : 'SAT',
                'v' : ['contrast', 'SAT'],
                't' : 'SAT',
                'a' : 'SAT'}
        inc = ['sv','sz','st']
        model_name = "M2"
        m = hddm.HDDM(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"])
    elif id > 7 and id < 12:
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1)
        ############## M3
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['SAT', 'givenResp'],
                'a' : 'SAT'}
        inc = ['sv','sz','st']
        model_name = "M3"
        m = hddm.HDDM(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz', 'st', "sz_SAT"])
    elif id > 11 and id < 16:
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1)
        ############## M4
        deps = {'sz' : 'SAT',
                'v' : ['SAT', 'contrast'],
                't' : ['SAT', 'givenResp'],
                'a' : 'SAT'}
        inc = ['sv','sz','st']
        model_name = "M4"
        m = hddm.HDDM(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz', 'st', "sz_SAT"])
    elif id > 15 and id < 20:
        dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
	dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
#        dataHDDM["abs_rt"] = np.abs(dataHDDM.rt)
#        dataHDDM["rt"] = dataHDDM.apply(lambda row: row["abs_rt"] if row['givenResp'] == 'Right' else -row["abs_rt"], axis=1)
        ############## M5
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : 'SAT',
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M5"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 19 and id < 24:
        dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
 #       dataHDDM["abs_rt"] = np.abs(dataHDDM.rt)
#        dataHDDM["rt"] = dataHDDM.apply(lambda row: row["abs_rt"] if row['givenResp'] == 'Right' else -row["abs_rt"], axis=1)
        ############## M6
        deps = {'sz' : 'SAT',
                'v' : ['SAT','contrast'],
                't' : 'SAT',
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M6"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")

    elif id > 23 and id < 28 :
        dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
 #       dataHDDM["abs_rt"] = np.abs(dataHDDM.rt)
 #       dataHDDM["rt"] = dataHDDM.apply(lambda row: row["abs_rt"] if row['givenResp'] == 'Right' else -row["abs_rt"], axis=1)
        ############## M7
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['SAT','givenResp'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M7"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 27:
	dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
 #       dataHDDM["abs_rt"] = np.abs(dataHDDM.rt)
 #       dataHDDM["rt"] = dataHDDM.apply(lambda row: row["abs_rt"] if row['givenResp'] == 'Right' else -row["abs_rt"], axis=1)
        ############## M8
        deps = {'sz' : 'SAT',
                'v' : ['SAT','contrast'],
                't' : ['SAT','givenResp'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M8"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")

    else :
        return np.nan()
    name = 'Exp2_depends_%s_%s' %(model_name, str(id))
    m.find_starting_values()
    m.sample(iter=20000, burn=18500, thin=1, dbname='DDM/traces/ModelSelection/db_%s'%name, db='pickle')
    m.save('DDM/Fits/ModelSelection/%s'%name)
    return m

v = ipyparallel.Client(profile="MS_exp2")[:]#sept
jobs = v.map(run_model, range(4 * 8))#4 chains for each model
wait_watching_stdout(jobs)
models = jobs.get()
