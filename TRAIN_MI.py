#%%training
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import sys
import os
import hyperparameters_mi as hyperparameters
from sklearn.metrics import roc_curve, roc_auc_score

#%% model hyperparameters 
rc_dropout = hyperparameters.rc_dropout
MI_dropout = hyperparameters.MI_dropout
#%% training hyperparameters 
schedual = hyperparameters.schedual
learning_rate = hyperparameters.learning_rate
num_epochs = hyperparameters.num_epochs
batch_size = hyperparameters.batch_size
#%% Functions 
path = hyperparameters.path
os.chdir(path)
from data import load_data, batch_generator_E1
from model_functions import naive_previous_basket, naive_freq_basket, top_prod_acc, basket_GRU, basket_GRU_MI, custom_BCE
#%%Load the data
BASE_RESULTS_FOLDER = hyperparameters.BASE_RESULTS_FOLDER
os.chdir(path+BASE_RESULTS_FOLDER)
trained_base_model = torch.load(hyperparameters.base_model_file)
base_model_state_dict = trained_base_model['model_state_dict']
base_model_last_vacc = trained_base_model['vacc'][-1]
hidden_dim = base_model_state_dict['W_fb'].size()[1]
f_name  = "MI" + str(hidden_dim)+"_lr"+str(learning_rate)+"_sl"+str(schedual)+".pth"
if hyperparameters.no_cross:
    f_name  = "NO_CROSS_MI" + str(hidden_dim)+"_lr"+str(learning_rate)+"_sl"+str(schedual)+".pth"



train_data      = trained_base_model['train_data']
train_customers = trained_base_model['train_customers']
val_data        = trained_base_model['val_data']
val_customers   = trained_base_model['val_customers']
test_data       = trained_base_model['test_data']
test_customers  = trained_base_model['test_customers']
price_data      = trained_base_model['price_data']
order_week_dict = trained_base_model['order_week_dict']

price_data[:,1,:] = price_data[:,1,:]/100 

discount_mask = torch.sum(price_data[:,1] > 0 ,dim =0) >= hyperparameters.week_th



print('Data is loaded')
train_generator = batch_generator_E1(train_data, price_data, train_customers, order_week_dict,batch_size)
val_generator   = batch_generator_E1(val_data, price_data, val_customers, order_week_dict,batch_size)
test_generator  = batch_generator_E1(test_data, price_data, test_customers, order_week_dict,batch_size)
del trained_base_model
#%% Baselines
ntime = train_data.size()[1]
nprod = train_data.size()[2]
base_model = basket_GRU(nprod,hidden_dim) 
base_model.load_state_dict(base_model_state_dict)
no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))
baseline1 = naive_previous_basket()
baseline2 = naive_freq_basket()
acc1 = 0
auc1 = 0
acc2 = 0
auc2 = 0
acc  = 0
auc  = 0

for i in np.arange(no_test_batches):
    (customers, inputs,targets,seq_lengths, output_prices) = next(test_generator)
    pred1 = baseline1(inputs)
    pred2 = baseline2(inputs)
    pred  = base_model(inputs)
    
    t_last = targets[torch.arange(targets.size()[0]), seq_lengths-1, :]
    pred1_last = pred1[torch.arange(pred1.size()[0]), seq_lengths-1, :]
    pred2_last = pred2[torch.arange(pred2.size()[0]), seq_lengths-1, :]
    pred_last = pred[torch.arange(pred.size()[0]), seq_lengths-1, :]
    
    auc1 += roc_auc_score(t_last.reshape(-1).detach().numpy(),pred1_last.reshape(-1).detach().numpy())
    auc2 += roc_auc_score(t_last.reshape(-1).detach().numpy(),pred2_last.reshape(-1).detach().numpy())
    auc  += roc_auc_score(t_last.reshape(-1).detach().numpy(),pred_last.reshape(-1).detach().numpy())
    
    
    

    acc1 += top_prod_acc(pred1,targets,seq_lengths)
    acc2 += top_prod_acc(pred2,targets,seq_lengths)
    acc += top_prod_acc(pred,targets,seq_lengths)


print(f"Baseline 1: acc = {acc1/no_test_batches:.3f}, auc = {auc1/no_test_batches:.3f}")
print(f"Baseline 2: acc = {acc2/no_test_batches:.3f}, auc = {auc2/no_test_batches:.3f}")
print(f"Base model: acc = {acc/no_test_batches:.3f}, auc = {auc/no_test_batches:.3f}")


#%% Initiate MI model 
network = basket_GRU_MI(nprod,hidden_dim,discount_mask,no_cross= hyperparameters.no_cross) 
state_dict = network.state_dict()
for weight in base_model_state_dict:
    state_dict[weight] = base_model_state_dict[weight]
network.load_state_dict(state_dict)
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
network = network.to(device)

#%% Initiate optimizer
only_last_val = True
only_last_train = False
criterion = custom_BCE(only_last_train)
val_criterion = custom_BCE(only_last_val)
optimizer = torch.optim.Adam(network.parameters(),lr=learning_rate) 

if type(schedual) != bool:
    lambda1 = lambda epoch: schedual ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

train_batches_per_epoch = np.ceil(train_data.size()[0]/batch_size)
val_batches_per_epoch = np.ceil(val_data.size()[0]/batch_size)
num_iter = num_epochs*train_batches_per_epoch

iter_loss = 0
iter_acc = 0
iter_n = 0
epoch_count = 0

hyper_dict = {'DATA_FOLDER': hyperparameters.DATA_FOLDER,
              'batch_size' : batch_size,
              'hidden_dim': hidden_dim,
              'rc_dropout':rc_dropout,
              'learning_rate':learning_rate,
              'num_epochs':num_epochs,
              'only_last_val': only_last_val,
              'only_last_train':only_last_train,
              'device':device,
              'optimizer':optimizer,
              'MI_dropout':MI_dropout}



train_loss = np.zeros(num_epochs)
val_loss = np.zeros(num_epochs)
val_auc = np.zeros(num_epochs)

train_acc = np.zeros(num_epochs)
val_acc = np.zeros(num_epochs)
state_dict_dict = {}
checkpoint = copy.deepcopy(network)
save_iter = 10

#%% Training
os.chdir(path + hyperparameters.MI_RESULTS_FOKDER)
begin_time = time.time()
start_time = time.time()
print("Start network training")
print(f"Training settings: Batch size = {batch_size}, Learning rate = {learning_rate}, Hidden dimension = {hidden_dim}, MI_dropout = {MI_dropout}")
for i in np.arange(num_iter):
    (index, inputs, targets, seq_lengths, output_prices) = next(train_generator)
    inputs = inputs.to(device)
    targets = targets.to(device)
    seq_lengths = seq_lengths.to(device)

    
    pred = network(inputs,output_prices,rc_dropout = rc_dropout,MI_dropout = MI_dropout)
    loss = criterion(pred,targets,seq_lengths)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = top_prod_acc(pred,targets,seq_lengths)
    iter_loss += inputs.size()[0]*loss.item()
    iter_acc += inputs.size()[0]*acc
    iter_n += inputs.size()[0]

    if ((i+1) % train_batches_per_epoch == 0):
        train_loss[epoch_count] = iter_loss/iter_n
        train_acc[epoch_count] = iter_acc/iter_n
        valloss = 0
        valauc = 0

        valacc = 0
        testloss = 0
        testacc = 0
        baselineacc = 0
        
        for j in np.arange(val_batches_per_epoch):
            (index, inputs, targets, seq_lengths, output_prices) = next(val_generator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            seq_lengths = seq_lengths.to(device)

            val_pred = network(inputs,output_prices)
            t_last = targets[torch.arange(targets.size()[0]), seq_lengths-1, :]
            val_pred_last = val_pred[torch.arange(val_pred.size()[0]), seq_lengths-1, :]
            
            loss = val_criterion(val_pred,targets,seq_lengths)
            valloss += inputs.size()[0]*loss.item()
            valacc += inputs.size()[0]*top_prod_acc(val_pred,targets,seq_lengths)
            valauc  += inputs.size()[0]*roc_auc_score(t_last.reshape(-1).detach().numpy(),val_pred_last.reshape(-1).detach().numpy())


            
            
        if (epoch_count == 0):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
        elif (valacc/val_data.size()[0] > np.max(val_acc[np.nonzero(val_acc)])):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
            
        val_loss[epoch_count] = valloss/val_data.size()[0]
        val_auc[epoch_count]  = valauc/val_data.size()[0]

        val_acc[epoch_count]  = valacc/val_data.size()[0]
        state_dict_dict[epoch_count] = network.state_dict()


        iter_loss = 0
        iter_acc = 0
        iter_n = 0
        epoch_count += 1


        expected_time_to_run = (time.time()-start_time) * (num_epochs - epoch_count) 
        expected_time_to_finish = time.time()+expected_time_to_run
        day_diff = int(time.strftime('%d', time.localtime(expected_time_to_finish))) - int(time.strftime('%d', time.localtime(time.time())))            
        expected_time_to_finish_format = time.strftime('%H:%M', time.localtime(expected_time_to_finish))
        publish_time = time.strftime('%H:%M', time.localtime(time.time()))
        if day_diff > 0:
            expected_time_to_finish_format = expected_time_to_finish_format+' (+'+str(day_diff) +')'
                    
        print("{} -> Epoch {}/{} ({:6} s), exp TTF {} : training loss = {:7}, validation loss = {:7}, training acc = {:7}, validation acc = {:7}, validation auc = {:7}, lr = {:7}".format(publish_time,
                                                                                                                                                                        epoch_count,
                                                                                                                                                                        num_epochs,
                                                                                                                                                                        np.round(time.time()-start_time,1),
                                                                                                                                                                        expected_time_to_finish_format,
                                                                                                                                                                        np.round(train_loss[epoch_count-1],6),
                                                                                                                                                                        np.round(val_loss[epoch_count-1],6),
                                                                                                                                                                        np.round(train_acc[epoch_count-1],4),
                                                                                                                                                                        np.round(val_acc[epoch_count-1],4),
                                                                                                                                                                        np.round(val_auc[epoch_count-1],4),
                                                                                                                                                                        np.round(optimizer.param_groups[0]["lr"],6)))
        if epoch_count % save_iter == 0:
            torch.save({'epoch': epoch_count,
                        'model_state_dict': checkpoint.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyper_dict':hyper_dict,
                        'vloss': val_loss,
                        'tloss': train_loss,
                        'vacc': val_acc,
                        'tacc': train_acc,
                        'test_data': test_data,
                        'test_customers': test_customers, 
                        'price_data': price_data,
                        'order_week_dict': order_week_dict,
                        'discount_mask': discount_mask
                        }, 'temp_' + f_name) # saves the network, losses and accuracies
            

        start_time = time.time()
        if type(schedual) != bool:
            scheduler.step()       

torch.save({'epoch': epoch_count,
            'model_state_dict': checkpoint.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyper_dict':hyper_dict,
            'vloss': val_loss,
            'tloss': train_loss,
            'vacc': val_acc,
            'tacc': train_acc,
            'test_data': test_data,
            'test_customers': test_customers, 
            'price_data': price_data,
            'order_week_dict': order_week_dict,
            'discount_mask': discount_mask
            }, f_name) # saves the network, losses and accuracies
#%% Test
no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))
test_acc = []
for i in np.arange(no_test_batches):
    (index, inputs, targets, seq_lengths, output_prices) = next(test_generator)
    pred = network(inputs,output_prices)
    test_acc.append(top_prod_acc(pred,targets,seq_lengths))


print("Finished training in ({:6} s), test acc = {:6} ({})".format(np.round(time.time()-begin_time,1),
                                                                   np.round(np.mean(test_acc),4),
                                                                   np.round(np.std(test_acc),4)))
