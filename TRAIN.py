#%%training
import numpy as np
import torch
import copy
import time
import os
import hyperparameters
#%% model hyperparameters 
hidden_dim = hyperparameters.hidden_dim
rc_dropout = hyperparameters.rc_dropout
#%% training hyperparameters 
schedual = hyperparameters.schedual
learning_rate = hyperparameters.learning_rate
num_epochs = hyperparameters.num_epochs
batch_size = hyperparameters.batch_size
f_name  = "basketGRU" + str(hidden_dim)+"_lr"+str(learning_rate)+"_sl"+str(schedual)+".pth"
#%%
DATA_FOLDER = hyperparameters.data_folder
results_folder = hyperparameters.results_folder

path = hyperparameters.path
split_vec = hyperparameters.split_vec
os.chdir(path)
from data import load_data, train_val_test_split, batch_generator
from model_functions import naive_previous_basket, naive_freq_basket, top_prod_acc, basket_GRU, custom_BCE
#%% Load data
customer_dict, order_dict, article_dict, order_week_dict, orderline_data, price_data = load_data(DATA_FOLDER)

print('Data is loaded')
train_data, train_customers, val_data, val_customers, test_data, test_customers  = train_val_test_split(orderline_data, split_vec, random = True)
train_batch_generator = batch_generator(train_data,batch_size,random = False)
val_batch_generator   = batch_generator(val_data,batch_size,random   = False)
test_batch_generator  = batch_generator(test_data,batch_size,random  = False)

#%% Baselines
no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))
baseline1 = naive_previous_basket()
baseline2 = naive_freq_basket()
acc1 = 0
acc2 = 0
for i in np.arange(no_test_batches):
    (index, inputs, targets, seq_lengths) = next(test_batch_generator)
    pred1 = baseline1(inputs)
    pred2 = baseline2(inputs)

    acc1 += top_prod_acc(pred1,targets,seq_lengths)
    acc2 += top_prod_acc(pred2,targets,seq_lengths)

print(f"Baseline 1: model = {baseline1}, acc = {acc1/no_test_batches}")
print(f"Baseline 2: model = {baseline2}, acc = {acc2/no_test_batches}")

#%% Initiate network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nprod = train_data.size()[2]
network = basket_GRU(nprod,hidden_dim) 
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
hyper_dict = {'DATA_FOLDER': DATA_FOLDER,
              'split_vec'  : split_vec,
              'batch_size' : batch_size,
              'hidden_dim': hidden_dim,
              'rc_dropout':rc_dropout,
              'learning_rate':learning_rate,
              'num_epochs':num_epochs,
              'only_last_val': only_last_val,
              'only_last_train':only_last_train,
              'device':device,
              'optimizer':optimizer,
              'schedual':schedual}
epoch_count = 0
train_loss = np.zeros(num_epochs)
val_loss = np.zeros(num_epochs)
train_acc = np.zeros(num_epochs)
val_acc = np.zeros(num_epochs)

checkpoint = copy.deepcopy(network)
save_iter = 10

#%% Training
os.chdir(path +results_folder)
begin_time = time.time()
start_time = time.time()
print("Start network training")
print(f"Training settings: Batch size = {batch_size}, Learning rate = {learning_rate}, Hidden dimension = {hidden_dim}, Dropout = {rc_dropout}, schedual = {schedual}")
for i in np.arange(num_iter):
    (index, inputs, targets, seq_lengths) = next(train_batch_generator)
    inputs = inputs.to(device)
    targets = targets.to(device)
    seq_lengths = seq_lengths.to(device)

    
    pred = network(inputs,rc_dropout = rc_dropout)
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
        valacc = 0
        testloss = 0
        testacc = 0
        
        for j in np.arange(val_batches_per_epoch):
            (index, inputs, targets, seq_lengths) = next(val_batch_generator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            seq_lengths = seq_lengths.to(device)

            val_pred = network(inputs)
            loss = val_criterion(val_pred,targets,seq_lengths)
            valloss += inputs.size()[0]*loss.item()
            valacc += inputs.size()[0]*top_prod_acc(val_pred,targets,seq_lengths)
            
        if (epoch_count == 0):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
        elif (valacc/val_data.size()[0] > np.max(val_acc[np.nonzero(val_acc)])):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
            
        val_loss[epoch_count] = valloss/val_data.size()[0]
        val_acc[epoch_count]  = valacc/val_data.size()[0]


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
                    
        print("{} -> Epoch {}/{} ({:6} s), exp TTF {} : training loss = {:7}, validation loss = {:7}, training acc = {:7}, validation acc = {:7}, lr = {:7}".format(publish_time,
                                                                                                                                                                        epoch_count,
                                                                                                                                                                        num_epochs,
                                                                                                                                                                        np.round(time.time()-start_time,1),
                                                                                                                                                                        expected_time_to_finish_format,
                                                                                                                                                                        np.round(train_loss[epoch_count-1],4),
                                                                                                                                                                        np.round(val_loss[epoch_count-1],4),
                                                                                                                                                                        np.round(train_acc[epoch_count-1],4),
                                                                                                                                                                        np.round(val_acc[epoch_count-1],4),
                                                                                                                                                                        np.round(optimizer.param_groups[0]["lr"],6)))

        start_time = time.time()
        if type(schedual) != bool:
            scheduler.step()
        if epoch_count % save_iter == 0:
            torch.save({'epoch': epoch_count,
                        'model_state_dict': checkpoint.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vloss': val_loss,
                        'tloss': train_loss,
                        'vacc': val_acc,
                        'tacc': train_acc,
                        'test_data': test_data,
                        'test_customers': test_customers, 
                        'price_data': price_data,
                        'order_week_dict': order_week_dict,
                        }, 'temp_' + f_name) # saves the network, losses and accuracies
            

            


torch.save({'epoch': epoch_count,
            'hyper_dict':hyper_dict,
            'model_state_dict': checkpoint.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vloss': val_loss,
            'tloss': train_loss,
            'vacc': val_acc,
            'tacc': train_acc,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'test_customers': test_customers, 
            'train_customers': train_customers,
            'val_customers': val_customers,
            'price_data': price_data,
            'order_week_dict': order_week_dict
            }, f_name) # saves the network, losses and accuracies
#%% Test
no_test_batches = int(np.ceil(test_data.size()[0]/batch_size))
test_acc = []
for i in np.arange(no_test_batches):
    (index, inputs, targets, seq_lengths) = next(test_batch_generator)
    pred = network(inputs)
    test_acc.append(top_prod_acc(pred,targets,seq_lengths))


print("Finished training in ({:6} s), test acc = {:6} ({})".format(np.round(time.time()-begin_time,1),
                                                                   np.round(np.mean(test_acc),4),
                                                                   np.round(np.std(test_acc),4)))
