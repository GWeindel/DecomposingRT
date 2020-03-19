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
    dataHDDM["SAT"] = dataHDDM.apply(lambda row: -0.5 if row['SAT'] == "Accuracy" else 0.5, axis=1)
    dataHDDM["givenResp"] = dataHDDM['response'].copy()
    dataHDDM["contrast"] = (dataHDDM.contrast - 0.07)*10
    dataHDDM['correct'] = dataHDDM.apply(lambda row: 1 if row["response"] == row["stim"] else 0, axis=1)
    dataHDDM["mt"] = dataHDDM["mt"] - dataHDDM.mt.mean()

    def v_link_func(x, data=dataHDDM):
        stim = (np.asarray(dmatrix('0 + C(s, [[1], [-1]])',
                               {'s': data.stim.ix[x.index]})))
        return x*stim

    if id < 4:
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == row["stim"] else 0, axis=1) #Accuracy coding
        dataHDDM["givenResp"] = dataHDDM.apply(lambda row: -0.5 if row['givenResp'] == 'Right' else 0.5, axis=1)
        ############## M1
        LM = ['t ~ mt + SAT + givenResp + SAT:givenResp']
	deps = {'sz' : 'SAT',
                'v' : ['SAT','contrast'],
                'a' : 'SAT'}
        inc = ['sv','sz','st']
        model_name = "JointM1"
    elif id > 3 and id < 8:
        dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1) #Response coding
        dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
        ############## M2
        LM = [{'model':'t ~ mt + SAT', 'link_func': lambda x: x} ,{'model':'v ~ SAT + contrast + SAT:contrast', 'link_func':v_link_func}]
        deps = {'sz' : 'SAT',
                'a' : 'SAT'}
        inc = ['z', 'sv','sz','st']
        model_name = "JointM2"
    else :
        return np.nan()
    name = 'Exp2_%s_%s' %(model_name, str(id))
    m = hddm.HDDMRegressor(dataHDDM, LM , depends_on = deps,
            include=inc, group_only_nodes=['sv', 'sz','st', "sz_SAT"], group_only_regressors=False, keep_regressor_trace=True)
    m.find_starting_values()
    m.sample(iter=20000, burn=18500, thin=1, dbname='DDM/traces/db_%s'%name, db='pickle')
    m.save('DDM/Fits/%s'%name)
    return m

v = ipyparallel.Client(profile="joint_exp2")[:]#sept
jobs = v.map(run_model, range(4 * 2))#4 chains for each model
wait_watching_stdout(jobs)
models = jobs.get()

