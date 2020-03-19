#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:29:31 2015

@author: weindel
"""
import os #handy system and path functions
from psychopy import core, visual, event, gui, data, monitors
import psychopy.logging as logging
import csv
from psychopy.constants import *
import ctypes #for parallel port

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

expName = 'GaborM2' 
expInfo = {'participant':''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()
expInfo['date'] = data.getDateStr() 
expInfo['expName'] = expName

filename = _thisDir + os.sep + u'Data' + os.sep + '%s_%s' %(expInfo['participant'], expInfo['date'])

 #An ExperimentHandler isn't essential but helps with data savin
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    dataFileName=filename)
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
dataFile = open(filename+'.csv', 'w')
logging.console.setLevel(logging.WARNING)

#Definition moniteur sujet
mon1 = monitors.Monitor('Sujet_EMG')
mon1.setDistance(100) #cm
mon1.setWidth(30) #cm
mon1.setSizePix([800, 600]) 
mon1.saveMon() #now when you launch Mon Center it should be there

#create a window to draw in
win = visual.Window(size=[800, 600], monitor=mon1, gamma=1.194, allowGUI=False, units='deg',fullscr=True)

class KeyResponse:
    def __init__(self):
        self.keys=[]
        self.corr=0
        self.rt=None
        self.clock=None
        self.num_trial = None
        self.num_block = None
port_gauche = [112]
port_droit = [104]
lptno = 0x379#parallel port adress

def send_trigger(code):
    ctypes.windll.inpout32.Out32(0xBCC8, code)
    core.wait(0.002)
    ctypes.windll.inpout32.Out32(0xBCC8, 0)
send_trigger(0)

fixation = visual.GratingStim(win=win, 
            mask='cross', size=0.2, 
            pos=[0,0], sf=0, color='black')

croix = visual.GratingStim(win=win, 
            mask='cross', size=0.2, 
            pos=[0,0], sf=0, color='black')

oeil = visual.GratingStim(win=win, 
            mask='circle', size=0.2, 
            pos=[0,0], sf=0, color='black')

gabor = visual.GratingStim(win,tex="sin",
            mask="gauss",texRes=256,  pos=[0,0],
            size=2.5, sf=[1.2,0], ori = 0, name='gabor')

pause=visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Prenez un petit temps de pause, reposez-vous les yeux.... puis appuyez sur n'importe quel bouton pour continuer.")

texte_fin=visual.TextStim(win, height = 0.5, ori=0, name='texte_fin',
    text=u"Fin de l'expérience. Merci d'attendre les instructions de l'expérimentateur..")

instructions = visual.TextStim(win, ori=0,height = 0.5, name='instructions', 
    text=u"Début de l'expérience [REC]")

RTTextMeanSub = visual.TextStim(win, 
                        units='norm',height = 0.1,
                        text='Mean : ...',color='white',autoLog = False)

PrecTextMeanSub = visual.TextStim(win, 
                        units='norm',height = 0.1, pos=[0, -0.5],
                        text='Mean : ...',color='white',autoLog = False)

conditionText = visual.TextStim(win, 
                        units='norm',height = 0.1, pos=[0, 0],
                        text='...',color='white',autoLog = False)

################################### INSTRUCTIONS
while True:
    instructions.draw()
    win.flip()
    instr_event= event.getKeys() #passé par l'expérimentateur
    if instr_event:
        break
    if event.getKeys(["escape"]): core.quit()



#############################TRIALS
#Trial Handler
trials=data.TrialHandler(nReps=1, method=u'sequential', 
    originPath=None, extraInfo=expInfo,
    trialList=data.importConditions('Stimuli'+ os.sep + expInfo['participant']+'.csv'))
thisTrial=trials.trialList[0]
if thisTrial!=None:
    for paramName in thisTrial.keys():
        exec(paramName+'=thisTrial.'+paramName)

#Init
trialClock = core.Clock()
counter = 0
block = 1
rep = KeyResponse()
rep.clock = core.Clock()

#Trial loop
sumRT= 0
currentRT=0
sumPrec = 0
currentPrec = 0
for thisTrial in trials:
    if thisTrial!=None:
        for paramName in thisTrial.keys():
            exec(paramName+'=thisTrial.'+paramName) 
#Pause
    if counter == 100 :
        meanRT = sumRT / counter
        meanPrec = float(sumPrec) / float(counter)
        block +=1
        pausePending = False
        FBPending = True
        continueRoutine = True
        while continueRoutine:
            event.clearEvents(eventType='keyboard')
            if FBPending:
                RTTextMeanSub.text= u"Votre temps de réaction moyen est de %i millisecondes" %(meanRT*1000)
                RTTextMeanSub.draw()
                PrecTextMeanSub.text=u"Votre taux de réponses correctes est de %i %%" %(meanPrec*100)
                PrecTextMeanSub.draw()
                win.flip()
                if event.getKeys(["escape"]): core.quit()
                if event.getKeys(["space"]):#seul l'experimentateur peut passer cet écran
                    FBPending = False
                    pausePending = True
                code1= ctypes.windll.inpout32.Inp32(lptno)
                if code1 in port_gauche :#Bypass
                    core.wait(0.2)
                    code2 = ctypes.windll.inpout32.Inp32(lptno)
                    if code2 in port_droit :
                        pausePending = True
                        FBPending = False
            if pausePending:
                pause.draw()
                win.flip()
                if event.getKeys(["escape"]): core.quit()
                lecture_port = ctypes.windll.inpout32.Inp32(0x379)
                if lecture_port  in port_gauche or lecture_port in port_droit:
                    bypassPause=0
                    meanRT=0
                    sumRT = 0
                    sumPrec = 0
                    meanPrec=0
                    counter = 0
                    win.flip()
                    pausePending = False
                    continueRoutine=False
    counter += 1
    while counter == 1:#Affichage de la condition
        event.clearEvents(eventType='keyboard')
        if condition == "P": 
            conditionText.text = u"Prétez attention à votre précision"
        elif condition == "V":
            conditionText.text = u"Prétez attention à votre vitesse"
        conditionText.draw()
        win.flip()
        core.wait(5)
        break
#composants boucle

    rep.num_trial = counter
    rep.num_block = block

#Fixation
    win.flip()
    core.wait(0.5) #ISI
    fixation.autoDraw = True
    win.flip()
    core.wait(0.5) #Fix - stim


#Definition contraste et côté de réponse
    gabor.contrast = 0.5
    if corr_ans == 'right':
        corr_pos = [0.7,0]
        incorr_pos = [-0.7,0]
    elif corr_ans == 'left' :
        corr_pos = [-0.7,0]
        incorr_pos = [0.7,0]

#boucle
    rep.status = NOT_STARTED
    gabor.status = STARTED
    continueRoutine = True
    while continueRoutine:
        if condition == "P":
            fixation.draw()
        if condition == "V":
            fixation.draw()
        if gabor.status == STARTED:
            gabor.contrast += contraste #manipulation symétrique du contraste
            gabor.pos=corr_pos
            gabor.draw()
            gabor.contrast -= contraste
            gabor.pos=incorr_pos
            gabor.draw()
            win.flip()
            send_trigger(int(trigger))
            gabor.status = NOT_STARTED
        if rep.status == NOT_STARTED:
            rep.status = STARTED
            rep.clock.reset() 
        if event.getKeys(["escape"]): core.quit()
        if  rep.status == STARTED:
            theseKeys=[]
            lecture_port = ctypes.windll.inpout32.Inp32(0x379)
            if lecture_port  in port_gauche: 
                send_trigger(100)
                theseKeys.append("left")
            elif lecture_port in port_droit: 
                send_trigger(200)
                theseKeys.append("right")
            if len(theseKeys) > 0: 
                if "escape" in theseKeys: 
                    win.close()
                    core.quit()
                rep.rt = rep.clock.getTime()
                win.flip()
                rep.keys=[]
                rep.keys = theseKeys[0] 
                if rep.keys == corr_ans:
                    rep.corr = 1
                else:
                    rep.corr = 0
                fixation.autoDraw=False
        #composants moyennes
                currentRT = rep.rt
                currentPrec = rep.corr
                sumPrec = sumPrec + currentPrec
                sumRT = sumRT + currentRT
    
        #Fin de boucle
                thisExp.addData('contraste', contraste)
                thisExp.addData('direction', corr_ans)
                thisExp.addData('keys',rep.keys)
                thisExp.addData('precision',rep.corr)
                thisExp.addData('rt',rep.rt)
                thisExp.addData('condition', condition)
                thisExp.addData('trial',rep.num_trial)
                thisExp.addData('block',rep.num_block) 
                thisExp.addData('trigger', trigger)
                thisExp.nextEntry()
                continueRoutine = False

    if event.getKeys(["escape"]): 
        win.close
        core.quit()


#routine fin
finPending = True
continueRoutine=True
while continueRoutine:
    if  finPending:
        core.wait(2.0)
        texte_fin.draw()
        win.flip()
        core.wait(5.0)
        finPending = False
    if not continueRoutine:  
        break
    continueRoutine = False 
    if event.getKeys(["escape"]): 
        core.quit()


win.close()
core.quit()