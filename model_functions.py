# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:29:21 2022

@author: Thijs de Lange
"""

import numpy as np
import torch
import torch.nn as nn
Look_for_cuda = False
if Look_for_cuda:
    Look_for_cuda  = torch.cuda.is_available()

def order_to_tensor(order_list, dim = 6000):
    order_list = list(order_list)
    tensor = torch.zeros(dim) 
    for a in order_list:
        tensor[a] = 1
    return tensor  

class naive_previous_basket(nn.Module):
  """
  Class for the basket GRU without covariates, as described by Van Maasakkers, Donkers en Fok (2022)
  """

  def __init__(self):

    super(naive_previous_basket,self).__init__()
  def forward(self, x):
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],x.size()[2])
    
    for t in np.arange(x.size()[1]):  
        xt = x[:,t,:] 
        next_basket = xt
        pred_sequence[:,t,:] = next_basket            
    return pred_sequence

class naive_freq_basket(nn.Module):
  """
  Class for the basket GRU without covariates, as described by Van Maasakkers, Donkers en Fok (2022)
  """

  def __init__(self):

    super(naive_freq_basket,self).__init__()
  def forward(self, x):
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],x.size()[2])
    freq_vector   = torch.zeros(x.size()[0],x.size()[2])
    for t in np.arange(x.size()[1]):  
        xt = x[:,t,:] 
        freq_vector += xt
        pred_sequence[:,t,:] = torch.sigmoid(freq_vector)          
    return pred_sequence

class custom_BCE(torch.nn.Module):
  """
  Class for computing the binary cross-entropy loss of model output
  """
  
  def __init__(self, only_last = True):
    """
    Initializes a custom_BCE isntance
    
    Arguments:
      only_last   whether BCE should be computed on the last baskets only, or average over the entire sequence
    """
    self.only_last = only_last
    super(custom_BCE,self).__init__()
  
  def forward(self,pred,target,last):
    """
    Calculates binary cross-entropy loss for model output
     
    Arguments:
      pred    batch of predictions of size [batch_size] x [seq length] x [input dim]
      target  batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [batch_size]
    """
    if self.only_last:
        pred_last_basket = pred[torch.arange(target.size()[0]),last-1,:]
        target_last_basket = target[torch.arange(target.size()[0]),last-1,:]
        loss = pred_last_basket.clone()
        loss[target_last_basket==False] = 1 - loss[target_last_basket==False]
    else:
        mask = torch.sum(target,2)>0 #select orders die daadwerkelijk gevuld zijn
        pred_sequence = pred[mask]
        target_sequence = target[mask]
        loss = pred_sequence.clone()
        loss[target_sequence==False] = 1 - loss[target_sequence==False]

    loss[loss<1e-30] = 1e-30
    final_loss = -torch.mean(torch.log(loss))
    return final_loss

def top_prod_acc(pred, target, last, return_binary = False):
    """
    Function to calculate the accuracy of the top ranked products by the model
    
    Arguments:
      pred    batch of predictions of size [batch_size] x [seq length] x [input dim]
      target  batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [batch_size]
    """
    pred = pred.cpu()
    target = target.cpu()
    last = last.cpu()
    pred_last_basket = pred[torch.arange(target.size()[0]),last-1,:]
    target_last_basket = target[torch.arange(target.size()[0]),last-1,:]
    target_items = target_last_basket.nonzero() 
    pred_ranking = torch.sort(pred_last_basket,dim=1,descending=True).indices
    acc = 0
    if return_binary:
        binary = torch.zeros(pred.size()[0],pred.size()[-1])
    for i in np.arange(0,pred.size()[0]):
        target_prods = target_items[target_items[:,0]==i,1]
        pred_prods = pred_ranking[i,0:target_prods.size()[0]]
        if return_binary:
            binary[i] = order_to_tensor(pred_prods,pred.size()[-1])
        diff = np.setdiff1d(target_prods.numpy(),pred_prods.numpy())
        acc += 1-(np.shape(diff)[0]/target_prods.size()[0])
    av_acc = acc/pred.size()[0]
    if return_binary:
        return av_acc, binary
    else:
        return av_acc   

def avg_rank(pred, targets, seq_lengths):
    last_targets = targets[torch.arange(targets.size()[0]),seq_lengths-1,:]
    last_pred    = pred[torch.arange(pred.size()[0]),seq_lengths-1,:]
    rank = torch.argsort(torch.argsort(last_pred,descending = True))
    return torch.sum((rank*last_targets))/torch.sum(last_targets)

class basket_GRU(nn.Module):


  def __init__(self,input_dim,num_hidden):

    super(basket_GRU,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Forget gate parameters
    self.W_fb = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fb.data)
    self.W_fa = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fa.data)
    self.e_f = nn.Parameter(torch.zeros(num_hidden))

    # Input modulator gate parameters
    self.W_ib = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_ib.data)
    self.W_ia = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ia.data)
    self.e_i = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_sb = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_sb.data)
    self.W_sa = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_sa.data)
    self.e_s = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ba = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ba.data)
    self.e_o = nn.Parameter(torch.zeros(input_dim))

  def forward(self, x, rc_dropout = False, track_hiddens = False):
    rc_dropout_rate = 0
    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    hidden = self.h_init.to(device) 
    rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            

    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    for t in np.arange(x.size()[1]):      
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        forget_gate = torch.sigmoid(torch.matmul(xt,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
        modulator_gate = torch.sigmoid(torch.matmul(xt,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
        candidate_gate = torch.tanh(torch.matmul(xt,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
        if rc_dropout != False:
            candidate_gate.to(device)
            rc_dropout_mask.to(device)
            candidate_gate = candidate_gate*rc_dropout_mask
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        if track_hiddens:
          hiddens[:,t,:] = hidden
        next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ba) + self.e_o)
        pred_sequence[:,t,:] = next_basket            
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence

class basket_GRU_MI(nn.Module):


  def __init__(self,input_dim,num_hidden,discount_mask,no_cross = False):

    super(basket_GRU_MI,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Forget gate parameters
    self.W_fb = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fb.data)
    self.W_fa = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fa.data)
    self.e_f = nn.Parameter(torch.zeros(num_hidden))

    # Input modulator gate parameters
    self.W_ib = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_ib.data)
    self.W_ia = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ia.data)
    self.e_i = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_sb = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_sb.data)
    self.W_sa = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_sa.data)
    self.e_s = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ba = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ba.data)
    self.e_o = nn.Parameter(torch.zeros(input_dim))
    
    # MI parameter
    self.no_cross = no_cross
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)
    if no_cross:
        self.E = nn.Parameter(torch.Tensor(input_dim))


  def forward(self, x, f, rc_dropout = False, track_hiddens = False, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    hidden = self.h_init.to(device) 
    rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    for t in np.arange(x.size()[1]):      
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        dt = f[:,t,1,:][:,self.mask]
        #dt = f[:,t,1,:]
        if self.no_cross:
            dt = f[:,t,1,:]
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate) 
        forget_gate = torch.sigmoid(torch.matmul(xt,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
        modulator_gate = torch.sigmoid(torch.matmul(xt,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
        candidate_gate = torch.tanh(torch.matmul(xt,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
        if rc_dropout != False:
            candidate_gate.to(device)
            rc_dropout_mask.to(device)
            candidate_gate = candidate_gate*rc_dropout_mask
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        if track_hiddens:
          hiddens[:,t,:] = hidden
        qt = torch.matmul(hidden,self.W_ba)
        qts = torch.sigmoid(qt)   
          
        if self.no_cross:
            At = (dt*self.E)*self.mask
        else:
            #At = torch.matmul((qts*dt)[:,self.mask],self.E)
            At = torch.matmul(dt,self.E)  
            if MI_dropout != False:
                At = At * MI_dropout_mask

        
        ### (dt*qts)*E
        # Wat lijkt opzich te werken: geen exp en wel qts in de laatste laag


        next_basket = torch.sigmoid(qt *(torch.ones(self.input_dim) + At) + self.e_o)
        #next_basket = torch.sigmoid(qt + qts*At + self.e_o)
        #next_basket = torch.sigmoid(qt + At + self.e_o)
        #next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ba) + At + self.e_o)

        pred_sequence[:,t,:] = next_basket            
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence
  
  def one_step(self,x,hidden,f):
      dt = f[:,1,:][:,self.mask]
      forget_gate = torch.sigmoid(torch.matmul(x,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
      modulator_gate = torch.sigmoid(torch.matmul(x,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
      candidate_gate = torch.tanh(torch.matmul(x,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
      hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
      At = torch.matmul(dt,self.E)  
      qt = torch.matmul(hidden,self.W_ba)
      next_basket = torch.sigmoid(qt *(torch.ones(self.input_dim) + At) + self.e_o)
      return next_basket



      
      
  
class basket_GRU_MI2(nn.Module):


  def __init__(self,input_dim,num_hidden,discount_mask):

    super(basket_GRU_MI2,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Forget gate parameters
    self.W_fb = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fb.data)
    self.W_fa = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fa.data)
    self.e_f = nn.Parameter(torch.zeros(num_hidden))

    # Input modulator gate parameters
    self.W_ib = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_ib.data)
    self.W_ia = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ia.data)
    self.e_i = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_sb = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_sb.data)
    self.W_sa = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_sa.data)
    self.e_s = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ba = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ba.data)
    self.e_o = nn.Parameter(torch.zeros(input_dim))
    
    # MI parameter
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)

  def forward(self, x, f, rc_dropout = False, track_hiddens = False, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    hidden = self.h_init.to(device) 
    rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    for t in np.arange(x.size()[1]):      
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        dt = f[:,t,1,:][:,self.mask]
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate) 
        forget_gate = torch.sigmoid(torch.matmul(xt,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
        modulator_gate = torch.sigmoid(torch.matmul(xt,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
        candidate_gate = torch.tanh(torch.matmul(xt,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
        if rc_dropout != False:
            candidate_gate.to(device)
            rc_dropout_mask.to(device)
            candidate_gate = candidate_gate*rc_dropout_mask
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        if track_hiddens:
          hiddens[:,t,:] = hidden
          
        At = torch.matmul(dt,self.E)        
        if MI_dropout != False:
            At = At * MI_dropout_mask
        qt = torch.matmul(hidden,self.W_ba)

        next_basket = torch.sigmoid(qt + At + self.e_o)
        #next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ba) + At + self.e_o)

        pred_sequence[:,t,:] = next_basket            
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence
  
class basket_GRU_MIS(nn.Module):


  def __init__(self,input_dim,discount_mask):

    super(basket_GRU_MIS,self).__init__()
    self.input_dim = input_dim

    self.e_o = nn.Parameter(torch.zeros(input_dim))

    # MI parameter
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)

  def forward(self, q, f, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    pred_sequence = torch.zeros(q.size()[0],q.size()[1],self.input_dim)
    for t in np.arange(q.size()[1]):      
        qt = q[:,t,:] # xt should have dimensions [batch_size, input_dim]
        dt = f[:,t,1,:][:,self.mask]
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate)           
        At = torch.matmul(dt,self.E)        
        if MI_dropout != False:
            At = At * MI_dropout_mask
        next_basket = torch.sigmoid(qt *(torch.ones(self.input_dim) + At) + self.e_o)
        pred_sequence[:,t,:] = next_basket            
    return pred_sequence

class basket_GRU_MIS2(nn.Module):


  def __init__(self,input_dim,discount_mask):

    super(basket_GRU_MIS2,self).__init__()
    self.input_dim = input_dim

    self.e_o = nn.Parameter(torch.zeros(input_dim))

    # MI parameter
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)

  def forward(self, q, f, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    pred_sequence = torch.zeros(q.size()[0],q.size()[1],self.input_dim)
    for t in np.arange(q.size()[1]):      
        qt = q[:,t,:] # xt should have dimensions [batch_size, input_dim]
        dt = f[:,t,1,:][:,self.mask]
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate)           
        At = torch.matmul(dt,self.E)        
        if MI_dropout != False:
            At = At * MI_dropout_mask
        next_basket = torch.sigmoid(qt + At + self.e_o)
        pred_sequence[:,t,:] = next_basket            
    return pred_sequence

class basket_GRU_MI_E(nn.Module):


  def __init__(self,input_dim,num_hidden,discount_mask):

    super(basket_GRU_MI_E,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # MI parameter
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)

    # Forget gate parameters
    self.W_fb = nn.Parameter(torch.Tensor(input_dim+self.discount_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fb.data)
    self.W_fa = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fa.data)
    self.e_f = nn.Parameter(torch.zeros(num_hidden))

    # Input information gate parameters
    self.W_ib = nn.Parameter(torch.Tensor(input_dim+self.discount_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_ib.data)
    self.W_ia = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ia.data)
    self.e_i = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_sb = nn.Parameter(torch.Tensor(input_dim+self.discount_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_sb.data)
    self.W_sa = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_sa.data)
    self.e_s = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ba = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ba.data)
    self.e_o = nn.Parameter(torch.zeros(input_dim))
    

  def forward(self, x, fo,fi, rc_dropout = False, track_hiddens = False, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    hidden = self.h_init.to(device) 
    rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    for t in np.arange(x.size()[1]):      
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        dit = fi[:,t,1,:][:,self.mask]
        dot = fo[:,t,1,:][:,self.mask]
        catvec = torch.cat((xt,dit),1)
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate) 
        forget_gate = torch.sigmoid(torch.matmul(catvec,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
        modulator_gate = torch.sigmoid(torch.matmul(catvec,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
        candidate_gate = torch.tanh(torch.matmul(catvec,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
        if rc_dropout != False:
            candidate_gate.to(device)
            rc_dropout_mask.to(device)
            candidate_gate = candidate_gate*rc_dropout_mask
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        if track_hiddens:
          hiddens[:,t,:] = hidden
          
        At = torch.matmul(dot,self.E)        
        if MI_dropout != False:
            At = At * MI_dropout_mask
        next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ba) *(torch.ones(self.input_dim) + At) + self.e_o)
        pred_sequence[:,t,:] = next_basket            
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence

class basket_GRU_MI_E2(nn.Module):


  def __init__(self,input_dim,num_hidden,discount_mask):

    super(basket_GRU_MI_E2,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Forget gate parameters
    self.W_fb = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fb.data)
    self.W_fa = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fa.data)
    self.e_f = nn.Parameter(torch.zeros(num_hidden))

    # Input modulator gate parameters
    self.W_ib = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_ib.data)
    self.W_ia = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ia.data)
    self.e_i = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_sb = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_sb.data)
    self.W_sa = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_sa.data)
    self.e_s = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ba = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ba.data)
    self.e_o = nn.Parameter(torch.zeros(input_dim))
    
    # MI parameter
    self.mask = discount_mask
    self.discount_dim = int(torch.sum(discount_mask).item())
    self.E = nn.Parameter(torch.Tensor(self.discount_dim,input_dim))
    nn.init.xavier_uniform_(self.E.data)


  def forward(self, x, fo, fi, rc_dropout = False, track_hiddens = False, MI_dropout = False):
    rc_dropout_rate = 0
    MI_dropout_rate = 0
    if MI_dropout != False:
        MI_dropout_rate = MI_dropout
    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
    device = torch.device("cuda" if Look_for_cuda else "cpu")    
    hidden = self.h_init.to(device) 
    rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    for t in np.arange(x.size()[1]):      
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        bt = xt *(1- fi[:,t,1,:])
        dt = fo[:,t,1,:][:,self.mask]
        MI_dropout_mask = torch.bernoulli(torch.ones(self.input_dim) - MI_dropout_rate)/(1-MI_dropout_rate) 
        forget_gate = torch.sigmoid(torch.matmul(bt,self.W_fb)      + torch.matmul(hidden,self.W_fa)                        + self.e_f)
        modulator_gate = torch.sigmoid(torch.matmul(bt,self.W_sb)   + torch.matmul(hidden,self.W_sa)                        + self.e_s)
        candidate_gate = torch.tanh(torch.matmul(bt,self.W_ib)      + torch.matmul(hidden*modulator_gate,self.W_ia)         + self.e_i)
        if rc_dropout != False:
            candidate_gate.to(device)
            rc_dropout_mask.to(device)
            candidate_gate = candidate_gate*rc_dropout_mask
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        if track_hiddens:
          hiddens[:,t,:] = hidden
          
        At = torch.matmul(dt,self.E)        
        if MI_dropout != False:
            At = At * MI_dropout_mask
        qt = torch.matmul(hidden,self.W_ba)


        next_basket = torch.sigmoid(qt *(torch.ones(self.input_dim) + At) + self.e_o)

        pred_sequence[:,t,:] = next_basket            
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence

