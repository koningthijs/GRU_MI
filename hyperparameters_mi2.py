# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:39:26 2022

@author: thijs.delange
"""

#%% model hyperparameters 
Look_for_cuda = False
rc_dropout = 0.5
MI_dropout = 0.3
#%% training hyperparameters 
schedual = 0.99
learning_rate = 0.001 #0.1 werkt niet
num_epochs = 120
batch_size = 100
#%% Dataset and trained model
DATA_FOLDER = ''
BASE_RESULTS_FOLDER = ''
MI_RESULTS_FOKDER = ''
path = ""
base_model_file = ''
split_vec = [0.6,0.2,0.2]
#%% Mask
week_th = 1