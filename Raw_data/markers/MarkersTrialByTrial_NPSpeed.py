# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:41:00 2016

@author: weindel
"""

import os
import string
import numpy as np
import pandas as pd

###############################################################################
columns = ['participant','contraste', 'condition', 'expdResp', 'trialType',
    'response','nbrA', 'CA','IA', 'rt', 'pmt','mt', 'Apmt','FlaTime',
    'ForceTime', 'chanEMG1', 'trial', 'givenResp','com','EMGtrial']
data = pd.DataFrame(columns=columns)

expe = 'PhD'

###############################################################################
os.chdir('/home/gabriel/ownCloud/PhD/Projets/SATTM/Data/markers/%s'%expe)
list_dir = os.listdir(os.curdir)
list_ = []
for f in list_dir:
    raw_ = pd.read_table(f, header=0, sep=',',skipinitialspace=True,skiprows=1)
    raw_['participant'] = f
    list_.append(raw_)
raw = pd.concat(list_, ignore_index=True)

raw.Position = [(float(x)*1000)/2048 for x in raw.Position]#converting samples to ms
raw= raw[raw.Type != 'Time 0']
raw.reset_index(drop=True,inplace=True)
raw.reset_index(drop=False,inplace=True)#create column index for np
raw = raw.replace(['EMG_L', 'EMG_R'],['Left','Right'])
raw = raw.replace(['Erg2', 'Erg1'],['Left','Right'])
raw = raw.replace(['100', '200'],['Left','Right'])
raw = raw.replace(['660', '550'],['Force_Onset','Force_Onset'])

raw.ix[raw.Description == 'EMG_Onset', 'Type']= 'EMG'

###############################################################
trials=[]
i = 0
j = 1

print 'processing ...'

'''
Using Pandas to delimit trial, then convert to numpy to perform calculations
'''
while i < len(raw)-1:
    if raw.Type.iloc[i] == 'New Segment' :
        if raw.participant.iloc[i] != raw.participant.iloc[i-1] and i > 0:
            print 'Done for %s, total trial : %i' %(raw.participant.iloc[i-1],j)
            j = 1
        i += 1
        participant,contraste, condition, expdResp, trialType, response,nbrA, \
        CA,IA, RT, PMT,MT, APMT,FlaTime, ForceT, chanEMG1, givenResp, com, \
        = np.repeat(np.nan, 18)
        EMGtrial = ''
        participant = raw.participant[i]
        idxSeg = i
        idxStim = raw[idxSeg:idxSeg+4].index[np.where(raw.Type[idxSeg:idxSeg+4]
            == 'Stimulus')][0]
        #idxSeg+4 car déjà arrivé qu'il y a plusieurs emg/force onset entre
        #le new segment et le stimulus
#==============================================================================
#       Mapping des conditions
#==============================================================================
        Trigger = list(raw.Description.iloc[idxStim])
        if Trigger[0]=='1':#SAT condition from trigger
            condition = 'Speed'
        else:
            condition = 'Accuracy'
        if Trigger[1]=='1':# Recovering expected response from trigger
            expdResp = 'Left'
        else:
            expdResp = 'Right'
        contraste = int(Trigger[2])# Recovering contrast level from trigger
#==============================================================================
#       Délimitation de l'essai + récupération nombre d'évauches EMG/Force
#==============================================================================
        while raw.Type.iloc[i] != 'New Segment':#Upper limit of the trial
            i += 1
            if i == len(raw):
                print 'Finished !'
                break
        idxUpperLim = i
        try :
            idxResp = raw[idxStim:idxUpperLim+1].index[np.where(raw.Type[
                idxStim:idxUpperLim+1] == 'Response')][0]
        except :
            trialType = 'NR'#NoResponse
        else:
            trial = raw[idxStim:idxResp+1].as_matrix()
            idxResp = np.where(raw.Type[idxStim:idxUpperLim+1]
                == 'Response')[0][0]
            nbrA = len(np.where(trial[:,2]=='EMG_Onset')[0])
            nbrF = len(np.where(trial[:,2]=='Force_Onset')[0])
            pos_stim = float(trial[0,3])
            pos_rep = float(trial[-1,3])
            if nbrF > 1:
                pos_Force = float(trial[np.where(trial[:,2]=='Force_Onset'),3]
                [0][0])
            else :
                pos_Force = np.nan
#==============================================================================
#        Position of the response EMG
#==============================================================================
            if nbrA != 0 :#All trials with one EMG activation
                if trial[idxResp,2] == expdResp:
                    response = 1
                    givenResp = trial[idxResp,2]
                else:
                    response = 0
                    givenResp = trial[idxResp,2]
                try :
                    posEMGR = float(trial[np.where((trial[:,2] =='EMG_Onset') &
                                         (trial[:,5] == givenResp)),3][0][-1])
                #Prend le dernier EMG **SUR LE MEME CANAL**
                except:#no EMG on the response channel
                    Warning ('No EMGR for participant n = %s on index %i'
                             %(participant, len(trials)))
                    posEMGR = np.nan
#==============================================================================
#               Computes MT and PMT
#==============================================================================
                MT = pos_rep-posEMGR
                PMT = posEMGR-pos_stim#temps entre stimulus et dernier EMG
                CA = len(trial[np.where((trial[:,2] =='EMG_Onset')
                    & (trial[:,5] == expdResp))])# Number of correct activation
                IA = len(trial[np.where((trial[:,2] =='EMG_Onset')
                    & (trial[:,5] != expdResp))])# Number of incorrect activation
#==============================================================================
#               TrialType + composants MA
#==============================================================================
                if nbrA >= 2:
                    posEMG1 = trial[np.where(trial[:,2]=='EMG_Onset'), 3][0][0]
                    chanEMG1 = trial[np.where(trial[:,2]=='EMG_Onset'), 5][0][0]
                    APMT = posEMG1 - pos_stim#Calcul du temps entre le stimulus
                    #et la première activité EMG
                    FlaTime = posEMGR - posEMG1 #temps entre le premier EMG et
                    #le dernier avant la réponse
                    trialType = 'MA'#Multiple Activity
#==============================================================================
#                  Loop entre les EMG pour compter les échanges entre canaux
#==============================================================================
                    com = 0
                    k = -1
                    Sides = []
                    EMGtrials = []
                    for chan in trial[np.where(trial[:,2]=='EMG_Onset'),5][0] :
                        Sides.append(chan)
                        if chan == expdResp :
                            EMGtrials.append('C')
                        else:
                            EMGtrials.append('I')
                        k += 1
                        if k >= 1 and k < len(trial[np.where(trial[:,2]==
                        'EMG_Onset'),5][0]):
                            if Sides[k] != Sides[k-1]:
                                com += 1
                    EMGtrial = string.join(EMGtrials)
                else:
                    trialType = 'SA'#Single Activity
                    APMT = PMT
                    if response == 1:
                        EMGtrial = 'C'
                    else :
                        EMGtrial = 'I'
                    com = 0
            else:
                trialType = 'UT'#Unmarked Trial
            ForceT = pos_rep - pos_Force#Calcul du Temps de Force
            RT = pos_rep-pos_stim
        trials.append([participant,contraste, condition, expdResp, trialType, \
                        response,nbrA, CA,IA, RT, PMT,MT, APMT,FlaTime, ForceT,
                        chanEMG1, j, givenResp, com, EMGtrial])
        j += 1
#        if len(trials)%totalTrial == 0:
#            print 'Done for %s' %participant
#            j = 1
    else:
        i += 1


data = pd.DataFrame(trials, columns=columns)
data.to_csv('/home/gabriel/ownCloud/PhD/Projets/SATTM/Data/markers/MRK_SAT_%s.csv'%expe,
            index=False)
