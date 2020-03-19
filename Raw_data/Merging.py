# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:22:34 2017

@author: gabriel
"""

import pandas as pd


#==============================================================================
# Collapsing experiments
#==============================================================================
exp1 = pd.read_csv('markers/MRK_SAT_M2.csv')
exp2 = pd.read_csv('markers/MRK_SAT_PhD.csv')

exp1['exp'] = 1
exp2['exp'] = 2

exp1['participant'] = exp1['participant'].replace(
    ['S2','s3','S4','S5','S6','S7','S9','S10','S12','S13','S14','S15','S16',
    'S17'],
    ['S1_1','S2_1','S3_1','S4_1','S5_1','S6_1','S7_1','S8_1','S9_1','S10_1',
    'S11_1','S12_1','S13_1','S14_1']) #Avoids duplicated subj_idx

exp2['participant'] = exp2['participant'].replace(
        ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13',
        'S14', 'S15','S16'], ['S1_2','S2_2','S3_2','S4_2','S5_2','S6_2','S7_2',
        'S8_2','S9_2','S10_2','S11_2','S12_2','S13_2','S14_2','S15_2','S16_2'])

exp1['contraste'] = exp1['contraste'].replace([1,2,3,4,5],[0.01,0.025,0.07,0.15,0.3])
exp2['contraste'] = exp2['contraste'].replace([1,2,3],[0.01,0.07,0.15])

df = pd.concat([exp1,exp2])
df = df.reset_index()

#==============================================================================
# Saving Data
#==============================================================================
df.drop([u'index', u'nbrA', u'CA', u'IA',
       u'Apmt', u'FlaTime', u'ForceTime', u'chanEMG1',
       u'com', u'EMGtrial'], axis=1, inplace=True)
df.to_csv('../Analysis/MRK_SAT.csv', index=False)
