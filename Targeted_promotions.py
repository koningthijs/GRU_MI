# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:28:40 2022

@author: thijs.delange
"""
import os

path = "G:/My Drive/THESIS/MODEL/PYTHON/CODE_ARTICLE_LEVEL"
os.chdir(path)

import scipy.stats as ss
from model_functions import naive_previous_basket, naive_freq_basket, top_prod_acc, basket_GRU, basket_GRU_MI, basket_GRU_MI2, basket_GRU_MI_E, basket_GRU_MIS, custom_BCE, avg_rank
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from data import load_data, batch_generator_E2
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import sys

batch_size = 100


# %% Load base model and data
customer_dict, order_dict, article_dict, order_week_dict, _, _ = load_data("6K_FINAL_ARTICLE_TOP_1000_NEW")

BASE_RESULTS_FOLDER = "/FINAL_MODELS"
base_model_file = 'basketGRU600_lr0.01_sl0.985.pth'
os.chdir(path+BASE_RESULTS_FOLDER)
trained_base_model = torch.load(base_model_file)
base_model_state_dict = trained_base_model['model_state_dict']
base_model_last_vacc = trained_base_model['vacc'][-1]
hidden_dim = base_model_state_dict['W_fb'].size()[1]
nprod = base_model_state_dict['W_fb'].size()[0]
W_ba = base_model_state_dict['W_ba']


train_data = trained_base_model['train_data']
train_customers = trained_base_model['train_customers']
val_data = trained_base_model['val_data']
val_customers = trained_base_model['val_customers']
test_data = trained_base_model['test_data']
test_customers = trained_base_model['test_customers']
price_data = trained_base_model['price_data']

order_week_dict = trained_base_model['order_week_dict']

price_data[:, 1, :] = price_data[:, 1, :]/100

print('Data is loaded')
del trained_base_model
# %% Load MI
MI_RESULTS_FOLDER = "/FINAL_MODELS"
MI_model_file = 'MI600_lr0.001_sl0.985.pth'
os.chdir(path+MI_RESULTS_FOLDER)
MI_base_model = torch.load(MI_model_file)
MI_model_state_dict = MI_base_model['model_state_dict']
discount_mask = MI_base_model['discount_mask']
MI_model = basket_GRU_MI(nprod, hidden_dim, discount_mask)
MI_model.load_state_dict(MI_model_state_dict)





#%% Conducting experiment 
# 5 batchgenerators are needed 
week = 105
baseline_generator = batch_generator_E2(test_data, price_data, test_customers, order_week_dict, batch_size)


#results allocation
base_pred = torch.zeros(test_data.size()[0], test_data.size()[1]-1, test_data.size()[2])
hiddens   = torch.zeros(test_data.size()[0], test_data.size()[1]-1, hidden_dim)


no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))

for i in np.arange(no_test_batches):

    
    (customers, inputs, targets, seq_lengths, output_prices, input_prices) = next(baseline_generator)
    pred,hid = MI_model(inputs, output_prices,track_hiddens = True)
    base_pred[i*batch_size:(i+1)*batch_size] = pred
    hiddens[i*batch_size:(i+1)*batch_size] = hid


#%% Latest before week 
week = 105
latest_order= []
for customer in test_customers:
    order = 0
    while order_week_dict[(customer,order)] < week:
        if (customer,order+1) not in order_week_dict.keys():
            order+=1
            break
        order += 1

    latest_order.append(order-1)
#%% General info discounts 
week = 105
discount_data = price_data[:,1]
# Articles that were ever promoted (431)
d_mask = torch.sum(discount_data,dim = 0) > 0
# Count promoweeks per article 
week_count = torch.sum(discount_data > 0, dim = 0)
# Make three groups based on promo frequency
mask1 = (week_count > 0) * (week_count < 3)   #124
mask2 = (week_count >= 3) * (week_count < 8)  #142
mask3 = (week_count >= 8)                     #165
# Average discounts per week in last 40 weeks which is 35
avg_discounts = int(np.mean(torch.sum(discount_data[-40:] > 0,dim = 1).detach().numpy()))
# number of promotions in selected week
num_promotions_selected_week = torch.sum(discount_data[week]>0)    
# Avg discount per article 
avg_discount = torch.nan_to_num(torch.sum(discount_data,dim = 0)/torch.sum(discount_data>0,dim = 0))    
# Weekly average discount
weekly_avg = torch.sum(discount_data,dim = 1)/torch.sum(discount_data>0,dim = 1)
#%% Comparison
one_step_hiddens = hiddens[torch.arange(test_data.size()[0]),torch.tensor(latest_order)-1]
one_step_baskets = test_data[torch.arange(test_data.size()[0]),torch.tensor(latest_order)]
fin = price_data[week]
f = fin.repeat(test_data.size()[0],1,1,)
base_pred = MI_model.one_step(one_step_baskets,one_step_hiddens,f)
f0 = torch.zeros(f.size()[0],f.size()[1],f.size()[2])  
pred0 = MI_model.one_step(one_step_baskets,one_step_hiddens,f0)
#%%
impact = base_pred-pred0
fin_impact = torch.sum(base_pred*(1-f[:,1])*f[:,0] - pred0*f[:,0])/torch.sum(base_pred*(1-f[:,1])*f[:,0])*100
week_discount_mask = fin[1]>0
direct_impact = impact[:,week_discount_mask]
#%% cost analysis
High_0 = pred0 > 0.05
positive_impact = impact > 0
#small_impact = impact < 0.02
cost_per_item = fin[1]*fin[0]
cost_per_item = cost_per_item.repeat(1200,1)
expected_discount_given = torch.sum(base_pred*cost_per_item)
expected_wasted_discount = torch.sum(base_pred*cost_per_item * High_0)
#%% Pred neg
week_d_mask = discount_data[week] > 0
mask_neg = torch.ones(test_data.size()[-1])*discount_mask - torch.ones(test_data.size()[-1])*week_d_mask
disc_neg_biased = avg_discount * mask_neg
correction_factor = (torch.sum(discount_data[week])/torch.sum(discount_data[week]>0)) /(torch.sum(disc_neg_biased)/torch.sum(disc_neg_biased>0)) 
f_neg = torch.zeros(2,test_data.size()[-1])
f_neg[1,:] = disc_neg_biased *correction_factor
f_neg = f_neg.repeat(test_data.size()[0],1,1,)

pred_neg = MI_model.one_step(one_step_baskets,one_step_hiddens,f_neg)

tar_impact = (pred_neg - pred0)*discount_mask
#%% find top products per customer
k = 3
target_discount_indices = torch.zeros(test_data.size()[0],k)
target_discount_impacts = torch.zeros(test_data.size()[0],k)
for customer in range(test_data.size()[0]):
    target_discount_indices[customer] = torch.topk(tar_impact[customer], k).indices
    target_discount_impacts[customer] = torch.topk(tar_impact[customer], k).values
    
#% find 2 random per customer
import random as rdn
posible_indices = []
for i in range(discount_mask.size()[0]):
    if mask_neg[i]:
        posible_indices.append(i)
random_discount_indices = torch.zeros(test_data.size()[0],k)
for customer in range(test_data.size()[0]):
    rdn.shuffle(posible_indices)
    random_discount_indices[customer] = torch.tensor(posible_indices[:k])
#%% Create financials

targeted_discounts = torch.tensor(f.detach().numpy())
random_discounts = torch.tensor(f.detach().numpy())
for cust in range(f.size()[0]):
    for i in range(k):
        product = int(target_discount_indices[cust,i].item())
        targeted_discounts[cust,1,product] = (disc_neg_biased *correction_factor)[product]
        product_r = int(random_discount_indices[cust,i].item())
        random_discounts[cust,1,product] = (disc_neg_biased *correction_factor)[product_r]
#%% pred targetted


def pbs(pred,indices):
    pass
    

pred_tar = MI_model.one_step(one_step_baskets,one_step_hiddens,targeted_discounts)
bs_tar = torch.sum(pred_tar)/pred_tar.size()[0]
spb_tar = torch.sum((1-targeted_discounts[:,1])*targeted_discounts[:,0] * pred_tar)/pred_tar.size()[0]

pred_ran = MI_model.one_step(one_step_baskets,one_step_hiddens,random_discounts)
bs_ran = torch.sum(pred_ran)/pred_ran.size()[0]
spb_ran = torch.sum((1-random_discounts[:,1])*random_discounts[:,0] * pred_ran)/pred_ran.size()[0]



print(f" Basket size      = {bs_ran:.2f} -> {bs_tar:.2f} ({(bs_tar-bs_ran)/bs_ran*100:.2f} %)")
print(f" Spend per basket = {spb_ran:.2f} -> {spb_tar:.2f} ({(spb_tar-spb_ran)/spb_ran*100:.2f} %)")




     

        
