Exp 1 contains all files necessary to replicate experiment for experiment 1, same for Exp 2.

Structure is as follows : 
- Stimuli contains pre-randomized trials for each participant
- gabor_Training.py is launched first to train participants on the task
- gabor.py is the main task
- files ending with pyc are the respective compiled script version

Disclaimer : 
1) Psychopy should be installed as standalone in order to run the experiment codes
2) some features of the code are setting specific (e.g. parallel port response keys) and may need some adaptation to your local setting. 
3) this code uses CRT refresh rate properties to control stimulus timing, different behavior should be expected when running on another screen/graphic card setting.
