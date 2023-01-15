# -*- coding: utf-8 -*-
import os

path = ""
os.chdir(path)
import scipy.stats as ss
from model_functions import naive_previous_basket, naive_freq_basket, top_prod_acc, basket_GRU, basket_GRU_MI, basket_GRU_MI2, basket_GRU_MI_E, basket_GRU_MIS,avg_rank, basket_GRU_MIS2
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from data import  batch_generator_E2
import numpy as np
import torch

batch_size = 100


# %% Load base model and data
BASE_RESULTS_FOLDER = ""
base_model_file = ''
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
MI_RESULTS_FOLDER = ""
MI_model_file = ''
os.chdir(path+MI_RESULTS_FOLDER)
MI_trained_model = torch.load(MI_model_file)
MI_model_state_dict = MI_trained_model['model_state_dict']
discount_mask = MI_trained_model['discount_mask']
MI_model = basket_GRU_MI(nprod, hidden_dim, discount_mask)
MI_model.load_state_dict(MI_model_state_dict)

# %% Load MI2
MI_RESULTS_FOLDER = ""
MI2_model_file = ''
os.chdir(path+MI_RESULTS_FOLDER)
MI2_trained_model = torch.load(MI2_model_file)
MI2_model_state_dict = MI2_trained_model['model_state_dict']
discount_mask = MI2_trained_model['discount_mask']
MI2_model = basket_GRU_MI2(nprod, hidden_dim, discount_mask)
MI2_model.load_state_dict(MI2_model_state_dict)
# %% Load MIS
MI_RESULTS_FOLDER = ""
MI_model_file = ''
os.chdir(path+MI_RESULTS_FOLDER)
MIS_base_model = torch.load(MI_model_file)
MIS_model_state_dict = MIS_base_model['model_state_dict']
discount_mask = MIS_base_model['discount_mask']
MIS_model = basket_GRU_MIS(nprod, discount_mask)
MIS_model.load_state_dict(MIS_model_state_dict)
# %% Load MIS2
MI_RESULTS_FOLDER = ""
MIS2_model_file = ''
os.chdir(path+MI_RESULTS_FOLDER)
MIS2_trained_model = torch.load(MIS2_model_file)
MIS2_model_state_dict = MIS2_trained_model['model_state_dict']
discount_mask = MIS2_trained_model['discount_mask']
MIS2_model = basket_GRU_MIS2(nprod, discount_mask)
MIS2_model.load_state_dict(MIS2_model_state_dict)
#%% Load MI_E
MIE_RESULTS_FOLDER = ""
MIE_model_file = ''
os.chdir(path+MIE_RESULTS_FOLDER)
MIE_base_model = torch.load(MIE_model_file)
MIE_model_state_dict = MIE_base_model['model_state_dict']
discount_mask = MIE_base_model['discount_mask']
MIE_model = basket_GRU_MI_E(nprod,hidden_dim,discount_mask)
MIE_model.load_state_dict(MIE_model_state_dict)
# %% Baselines
nprod = train_data.size()[2]
base_model = basket_GRU(nprod, hidden_dim)
base_model.load_state_dict(base_model_state_dict)
no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))


no_MI_model = basket_GRU_MI(nprod, hidden_dim, discount_mask)
MI_model_state_dict['E'] = torch.zeros(MI_model_state_dict['E'].size()[
                                       0], MI_model_state_dict['E'].size()[1])
no_MI_model.load_state_dict(MI_model_state_dict)

baseline1 = naive_previous_basket()
baseline2 = naive_freq_basket()
# %% Make predictions
print('Make final preditions')
test_generator = batch_generator_E2(
    test_data, price_data, test_customers, order_week_dict, batch_size)


MIE_model_pred_last = torch.zeros(test_data.size()[0],test_data.size()[2])
MIE_model_bin_last = torch.zeros(test_data.size()[0],test_data.size()[2])

MIS2_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
MIS2_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

MIS_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
MIS_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

no_MI_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
no_MI_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

MI_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
MI_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

MI2_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
MI2_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

base_model_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
base_model_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

B1_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
B1_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

B2_pred_last = torch.zeros(test_data.size()[0], test_data.size()[2])
B2_bin_last = torch.zeros(test_data.size()[0], test_data.size()[2])

targets_last = torch.zeros(test_data.size()[0], test_data.size()[2])
seq_last = torch.zeros(test_data.size()[0], dtype=torch.long)
discount_last = torch.zeros(test_data.size()[0], test_data.size()[2])

no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))

acc_B1 = []
acc_B2 = []
acc_Base = []
acc_MI = []
acc_MI2 = []
acc_no_MI = []
acc_MIS = []
acc_MIS2 = []
acc_MIE = []

rank_B1 = []
rank_B2 = []
rank_Base = []
rank_MI = []
rank_MI2 = []
rank_no_MI = []
rank_MIS = []
rank_MIS2 = []
rank_MIE = []


AUC_B1 = []
AUC_B2 = []
AUC_Base = []
AUC_MI = []
AUC_MI2 = []
AUC_no_MI = []
AUC_MIS = []
AUC_MIS2 = []
AUC_MIE = []

APS_B1 = []
APS_B2 = []
APS_Base = []
APS_MI = []
APS_MI2 = []
APS_no_MI = []
APS_MIS = []
APS_MIS2 = []
APS_MIE = []


for i in np.arange(no_test_batches):
    (customers, inputs, targets, seq_lengths,
     output_prices, input_prices) = next(test_generator)
    pred1 = baseline1(inputs)
    pred2 = baseline2(inputs)
    pred_base, hiddens = base_model(inputs, track_hiddens=True)
    pred_MI = MI_model(inputs, output_prices)
    pred_MI2 = MI2_model(inputs, output_prices)

    pred_MIE = MIE_model(inputs, output_prices, input_prices)

    pred_no_MI = no_MI_model(inputs, output_prices)

    q = torch.matmul(hiddens, W_ba).detach()
    pred_MIS = MIS_model(q, output_prices)
    pred_MIS2 = MIS2_model(q, output_prices)
    discount = output_prices[:, :, 1, :]

    t_last = targets[torch.arange(targets.size()[0]), seq_lengths-1, :]
    d_last = discount[torch.arange(targets.size()[0]), seq_lengths-1, :]


    MIE_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_MIE[torch.arange(pred_MIS.size()[0]),seq_lengths-1,:]
    MIS_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_MIS[torch.arange(
        pred_MIS.size()[0]), seq_lengths-1, :]
    MIS2_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_MIS2[torch.arange(
        pred_MIS2.size()[0]), seq_lengths-1, :]
    no_MI_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_no_MI[torch.arange(
        pred_no_MI.size()[0]), seq_lengths-1, :]
    MI_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_MI[torch.arange(
        pred_MI.size()[0]), seq_lengths-1, :]
    MI2_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_MI2[torch.arange(
        pred_MI2.size()[0]), seq_lengths-1, :]
    base_model_pred_last[i*batch_size:(i+1)*batch_size] = pred_base[torch.arange(
        pred_base.size()[0]), seq_lengths-1, :]
    B1_pred_last[i*batch_size:(i+1)*batch_size] = pred1[torch.arange(
        pred1.size()[0]), seq_lengths-1, :]
    B2_pred_last[i*batch_size:(i+1)*batch_size] = pred2[torch.arange(
        pred2.size()[0]), seq_lengths-1, :]

    targets_last[i*batch_size:(i+1)*batch_size] = t_last
    discount_last[i*batch_size:(i+1)*batch_size] = d_last
    seq_last[i*batch_size:(i+1)*batch_size] = seq_lengths

    acc, binary = top_prod_acc(pred_MIE,targets,seq_lengths, return_binary= True)
    MIE_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_MIE.append(acc)

    acc, binary = top_prod_acc(pred1, targets, seq_lengths, return_binary=True)
    B1_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_B1.append(acc)

    acc, binary = top_prod_acc(pred2, targets, seq_lengths, return_binary=True)
    B2_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_B2.append(acc)

    acc, binary = top_prod_acc(
        pred_base, targets, seq_lengths, return_binary=True)
    base_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_Base.append(acc)

    acc, binary = top_prod_acc(
        pred_MI, targets, seq_lengths, return_binary=True)
    MI_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_MI.append(acc)
    
    acc, binary = top_prod_acc(
        pred_MI2, targets, seq_lengths, return_binary=True)
    MI2_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_MI2.append(acc)

    acc, binary = top_prod_acc(
        pred_no_MI, targets, seq_lengths, return_binary=True)
    no_MI_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_no_MI.append(acc)

    acc, binary = top_prod_acc(
        pred_MIS, targets, seq_lengths, return_binary=True)
    MIS_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_MIS.append(acc)
    
    acc, binary = top_prod_acc(
        pred_MIS2, targets, seq_lengths, return_binary=True)
    MIS2_model_bin_last[i*batch_size:(i+1)*batch_size] = binary
    acc_MIS2.append(acc)

    rank_B1.append(avg_rank(pred1, targets, seq_lengths))
    rank_B2.append(avg_rank(pred2, targets, seq_lengths))
    rank_Base.append(avg_rank(pred_base, targets, seq_lengths))
    rank_MI.append(avg_rank(pred_MI, targets, seq_lengths))
    rank_MI2.append(avg_rank(pred_MI2, targets, seq_lengths))

    rank_no_MI.append(avg_rank(pred_no_MI, targets, seq_lengths))
    rank_MIS.append(avg_rank(pred_MIS, targets, seq_lengths))
    rank_MIS2.append(avg_rank(pred_MIS2, targets, seq_lengths))

    rank_MIE.append(avg_rank(pred_MIE,targets,seq_lengths))

    AUC_B1.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                  pred1[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_B2.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                  pred2[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_Base.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                    pred_base[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_MI.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                  pred_MI[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_MI2.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                  pred_MI2[torch.arange(pred_MI2.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_no_MI.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                     pred_no_MI[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_MIS.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                   pred_MIS[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_MIS2.append(roc_auc_score(t_last.reshape(-1).detach().numpy(),
                   pred_MIS2[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    AUC_MIE.append(roc_auc_score(t_last.reshape(-1).detach().numpy(), pred_MIE[torch.arange(pred_MI.size()[0]),seq_lengths-1,:].reshape(-1).detach().numpy()))
    
    
    
    
    APS_B1.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                  pred1[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_B2.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                  pred2[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_Base.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                    pred_base[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_MI.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                  pred_MI[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_MI2.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                  pred_MI2[torch.arange(pred_MI2.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_no_MI.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                     pred_no_MI[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_MIS.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                   pred_MIS[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_MIS2.append(average_precision_score(t_last.reshape(-1).detach().numpy(),
                   pred_MIS2[torch.arange(pred_MI.size()[0]), seq_lengths-1, :].reshape(-1).detach().numpy()))
    APS_MIE.append(average_precision_score(t_last.reshape(-1).detach().numpy(), pred_MIE[torch.arange(pred_MI.size()[0]),seq_lengths-1,:].reshape(-1).detach().numpy()))



print(f"Baseline 1:  acc = {np.mean(acc_B1):.4f}  ({np.std(acc_B1)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_B1):.4f} ({np.std(rank_B1)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_B1):.4f} ({np.std(AUC_B1)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_B1):.4f} ({np.std(APS_B1)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"Baseline 2:  acc = {np.mean(acc_B2):.4f}  ({np.std(acc_B2)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_B2):.4f}  ({np.std(rank_B2)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_B2):.4f} ({np.std(AUC_B2)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_B2):.4f} ({np.std(APS_B2)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"Base model:  acc = {np.mean(acc_Base):.4f}  ({np.std(acc_Base)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_Base):.4f}  ({np.std(rank_Base)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_Base):.4f} ({np.std(AUC_Base)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_Base):.4f} ({np.std(APS_Base)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"MI        :  acc = {np.mean(acc_MI):.4f}  ({np.std(acc_MI)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_MI):.4f}  ({np.std(rank_MI)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_MI):.4f} ({np.std(AUC_MI)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_MI):.4f} ({np.std(APS_MI)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"MI2       :  acc = {np.mean(acc_MI2):.4f}  ({np.std(acc_MI2)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_MI2):.4f}  ({np.std(rank_MI2)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_MI2):.4f} ({np.std(AUC_MI2)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_MI2):.4f} ({np.std(APS_MI2)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"MIE       :  acc = {np.mean(acc_MIE):.4f}  ({np.std(acc_MIE)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_MIE):.4f}  ({np.std(rank_MIE)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_MIE):.4f} ({np.std(AUC_MIE)/np.sqrt(len(acc_B1)):.4f}), APS = {np.mean(APS_MIE):.4f} ({np.std(APS_MIE)/np.sqrt(len(acc_B1)):.4f}) ")
print(f"MIS       :  acc = {np.mean(acc_MIS):.4f}  ({np.std(acc_MIS)/np.sqrt(len(acc_B1)):.4f}),  rank = {np.mean(rank_MIS):.4f}  ({np.std(rank_MIS)/np.sqrt(len(acc_B1)):.4f}), AUC = {np.mean(AUC_MIS):.4f} ({np.std(AUC_MIS)/np.sqrt(len(acc_B1)):.4f}) ")

#%% Distribution plots
import seaborn as sns
import pandas as pd


no_knowledge_basline = torch.sum(targets_last)/(targets_last.size()[0]*targets_last.size()[1])

np.random.seed(10)

remove_n = 1

final_predictions = pd.DataFrame()
final_predictions['targets'] = targets_last.reshape(-1).detach().numpy()
final_predictions['B1 prediction'] = B1_pred_last.reshape(-1).detach().numpy()
final_predictions['B2 prediction'] = B2_pred_last.reshape(-1).detach().numpy()
final_predictions['Base model prediction'] = base_model_pred_last.reshape(-1).detach().numpy()
final_predictions['GRU+MI prediction'] = MI_model_pred_last.reshape(-1).detach().numpy()
final_predictions['Average'] = (final_predictions['GRU+MI prediction'] + final_predictions['Base model prediction'])/2
final_predictions['Frac'] = (final_predictions['GRU+MI prediction'] / final_predictions['Base model prediction'])


def categorise(row):  
    if row['targets'] == 0:
        return 'No purchase'
    elif row['targets'] == 1:
        return 'Purchase'

def greater_than(row):  
    if row['GRU+MI prediction'] >= row['Base model prediction']:
        return 'GRU+M.I.' 
    else:
        return 'Base model'
    

final_predictions['Actual'] = final_predictions.apply(lambda row: categorise(row), axis=1)
final_predictions['Dominant'] = final_predictions.apply(lambda row: greater_than(row), axis=1)



remove_n = int(len(final_predictions[final_predictions.targets == 0]) - sum(final_predictions['targets']))
drop_indices = np.random.choice(final_predictions[final_predictions.targets == 0].index, remove_n, replace=False)
final_predictions_skew_corrected = final_predictions.drop(drop_indices)
palette = sns.color_palette('RdGy')
palette = ['grey','red']
#%%
#sns.set(rc={'axes.facecolor':'white'})
sns.set_style("whitegrid")
ax = sns.histplot(final_predictions_skew_corrected, x="Base model prediction", hue="Actual",palette=palette,log_scale=True,element="step")
sns.move_legend(ax, "upper left")
ax.set(xlim=(1/10000000,1))
ax.set(ylim=(0,800))
ax.axvline(no_knowledge_basline,c = 'black',linewidth = 1)
ax.set(title='Histogram base model preditions')
sns.set(rc={'figure.figsize':(110.7,8.27)})

#%%
#sns.set(rc={'axes.facecolor':'white'})
sns.set_style("whitegrid")
ax = sns.histplot(final_predictions_skew_corrected, x="Base model prediction", hue="Actual",palette=palette,element="step")
sns.move_legend(ax, "upper left")
ax.set(xlim=(0,1))
ax.set(ylim=(0,5000))
ax.set(title='Histogram base model preditions')
sns.set(rc={'figure.figsize':(110.7,8.27)})


#%%
sns.set_style("whitegrid")

ax = sns.histplot(final_predictions_skew_corrected, x="GRU+MI prediction", hue="Actual",palette=palette,log_scale=True,element="step")
sns.move_legend(ax, "upper left")
ax.set(xlim=(1/10000000,1))
ax.set(ylim=(0,800))

ax.axvline(no_knowledge_basline,c = 'black',linewidth = 1)
ax.set(title='Histogram GRU+MI model preditions')
sns.set(rc={'figure.figsize':(110.7,8.27)})

#%%
sns.set_style("whitegrid")

ax = sns.histplot(final_predictions_skew_corrected, x="GRU+MI prediction", hue="Actual",palette=palette,element="step")
sns.move_legend(ax, "upper left")
ax.set(xlim=(0,1))
ax.set(ylim=(0,5000))

ax.set(title='Histogram GRU+MI model preditions')
sns.set(rc={'figure.figsize':(110.7,8.27)})
#%%
plt.tight_layout()
sns.displot(final_predictions_skew_corrected, x="GRU+MI prediction",hue = 'Actual', element="step",palette=palette,log_scale=True)
plt.title('Histogram GRU+MI  model preditions')
plt.xlim(1/1000000, 1)
plt.tight_layout()

    
    

#%%
import random
purchases = final_predictions_skew_corrected[final_predictions_skew_corrected.targets == 1]
no_purchases = final_predictions_skew_corrected[final_predictions_skew_corrected.targets == 0]


fig, ax = plt.subplots()
sns.set(rc={'figure.figsize':(7,6)})
sns.set_style("whitegrid")

sns.ecdfplot(purchases['Base model prediction'], ax = ax,label = 'Base model',log_scale=False,stat='proportion')
sns.ecdfplot(purchases['GRU+MI prediction'], ax = ax, label = 'GRU+MI model',log_scale=False,stat='proportion')
plt.xlim(-0.01, 0.2)
plt.legend()
plt.title('Emperical CDF for both the base model and the GRU+MI model')
plt.xlabel('Prediction')

# %% effect on promo articles
mask = discount_last > 0
discount_pred_MI = MI_model_pred_last[mask]
discount_pred_no_MI = no_MI_model_pred_last[mask]
diff = discount_pred_MI - discount_pred_no_MI
basket_size_increase = torch.sum(
    MI_model_pred_last-no_MI_model_pred_last)/torch.sum(MI_model_pred_last)*100
# %% Paired_t_test


def paired_t_test(x, y):
    n = len(x)
    df = (n-1)
    cv = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
          2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131]
    d = np.subtract(x, y)
    s = np.sum(d)
    t = s/np.sqrt((n*np.sum(d**2) - s**2)/(n-1))
    if df > len(cv):
        return t, np.abs(t) > 2
    else:
        return t, np.abs(t) > cv[df]


# %% acc
acc_list = [acc_B1, acc_B2, acc_Base, acc_MI, acc_MIS]
acc_mat = torch.zeros(len(acc_list), len(acc_list))

for xi in range(len(acc_list)):
    x = acc_list[xi]
    for yi in range(len(acc_list)):
        y = acc_list[yi]
        _, sig = paired_t_test(x, y)
        if sig == False:
            acc_mat[xi, yi] = 0
        else:
            acc_mat[xi, yi] = 1
print(acc_mat)

# %% rank
rank_list = [rank_B1, rank_B2, rank_Base, rank_MI, rank_MIS]
rank_mat = torch.zeros(len(rank_list), len(rank_list))

for xi in range(len(rank_list)):
    x = rank_list[xi]
    for yi in range(len(rank_list)):
        y = rank_list[yi]
        _, sig = paired_t_test(x, y)
        if sig == False:
            rank_mat[xi, yi] = 0
        else:
            rank_mat[xi, yi] = 1
print(rank_mat)

# %% AUC
AUC_list = [AUC_B1, AUC_B2, AUC_Base, AUC_MI, AUC_MIS]
AUC_mat = torch.zeros(len(AUC_list), len(AUC_list))

for xi in range(len(AUC_list)):
    x = AUC_list[xi]
    for yi in range(len(AUC_list)):
        y = AUC_list[yi]
        _, sig = paired_t_test(x, y)
        if sig == False:
            AUC_mat[xi, yi] = 0
        else:
            AUC_mat[xi, yi] = 1
print(AUC_mat)
# %% Per article performance


def per_dim_analysis(pred, target, dim=0):
    TP = torch.sum(pred*target, dim=dim)
    FP = torch.sum(pred*(1-target), dim=dim)
    TN = torch.sum((1-pred)*(1-target), dim=dim)
    FN = torch.sum((1-pred)*target, dim=dim)
    acc = (TP + TN)/(TP + FP + TN + FN)
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2*prec*recall)/(prec+recall)
    return TP, FP, TN, FN, acc, prec, recall, F1


# %% AUC per article bucket
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
# pylab.rcParams.update(params)
no_buckets = 5
bucket_size = int(np.round(test_data.size()[-1]/no_buckets))
global_sales = torch.sum(torch.sum(test_data, dim=0), dim=0)
total_sales = torch.sum(targets_last)
article_rank = torch.tensor(ss.rankdata(global_sales))
article_masks = {}
B1_pred_last
per_bucket_auc_B1 = []
per_bucket_auc_B2 = []
per_bucket_auc_Base = []
per_bucket_auc_MI = []

per_bucket_acc_B1 = []
per_bucket_acc_B2 = []
per_bucket_acc_Base = []
per_bucket_acc_MI = []

sales_percentage = []
for bucket in range(no_buckets):
    mask = (article_rank >= (bucket * bucket_size)) * \
        (article_rank < ((bucket+1) * bucket_size))
    article_masks[bucket] = mask
    tar = targets_last[:, mask].reshape(-1).detach().numpy()
    TP, FP, TN, FN, acc, prec, recall, F1 = per_dim_analysis(
        B1_bin_last, targets_last)
    per_bucket_acc_B1.append(np.nanmean(prec[mask]))
    TP, FP, TN, FN, acc, prec, recall, F1 = per_dim_analysis(
        B2_bin_last, targets_last)
    per_bucket_acc_B2.append(np.nanmean(prec[mask]))
    TP, FP, TN, FN, acc, prec, recall, F1 = per_dim_analysis(
        base_model_bin_last, targets_last)
    per_bucket_acc_Base.append(np.nanmean(prec[mask]))
    TP, FP, TN, FN, acc, prec, recall, F1 = per_dim_analysis(
        MI_model_bin_last, targets_last)
    per_bucket_acc_MI.append(np.nanmean(prec[mask]))

    sales_percentage.append(np.round((sum(tar)/total_sales).item()*100, 1))
    per_bucket_auc_B1.append(roc_auc_score(
        tar, B1_pred_last[:, mask].reshape(-1).detach().numpy()))
    per_bucket_auc_B2.append(roc_auc_score(
        tar, B2_pred_last[:, mask].reshape(-1).detach().numpy()))
    per_bucket_auc_Base.append(roc_auc_score(
        tar, base_model_pred_last[:, mask].reshape(-1).detach().numpy()))
    per_bucket_auc_MI.append(roc_auc_score(
        tar, MI_model_pred_last[:, mask].reshape(-1).detach().numpy()))


labels = [str(i+1) + ' ('+str(sales_percentage[i]) +
          '%)' for i in range(no_buckets)]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, per_bucket_auc_B1, width,
                label='B1', color='grey', alpha=0.5)
rects2 = ax.bar(x - .5 * width, per_bucket_auc_B2,
                width, label='B2', color='grey')
rects3 = ax.bar(x + .5 * width, per_bucket_auc_Base, width,
                label='Base', color='red', alpha=0.5)
rects4 = ax.bar(x + 1.5 * width, per_bucket_auc_MI,
                width, label='GRU+MI', color='red')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC - scores')
ax.set_xlabel('Buckets')

ax.set_title('AUC - score per article bucket based on popularity')
ax.set_xticks(x, labels)
ax.legend(loc=3)

ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
ax.bar_label(rects3, padding=3, fmt='%.2f')
ax.bar_label(rects4, padding=3, fmt='%.2f')


fig.tight_layout()


plt.show()



