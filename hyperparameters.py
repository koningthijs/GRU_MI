# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:39:26 2022

@author: thijs.delange
"""

#%% model hyperparameters 
Look_for_cuda = False
hidden_dim = 900
rc_dropout = 0.5
#%% training hyperparameters 
schedual = 0.99
learning_rate = 0.01 #0.1 werkt niet
num_epochs = 120
batch_size = 100
#%% Dataset
data_folder = ""            # Path to where the data is stored
results_folder = ''         # Folder to store the trained model
path = ""                   # general path to folder in which the model_functions file,TRAIN file and data file are stored in. 
split_vec = [0.6,0.2,0.2]