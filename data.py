# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import os
import pandas as pd 
import torch
import numpy as np
import torch.nn as nn
import random as rdn
data_path = ""

#%% progress bar 
import sys
import re


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
#%%
def order_to_tensor(order_list, dim = 6000):
    order_list = list(order_list)
    tensor = torch.zeros(dim) 
    for a in order_list:
        tensor[a] = 1
    return tensor  

def load_data(path):
    os.chdir(data_path+"/"+path)
    if os.path.exists("orderline_data.pt")  and os.path.exists("article_dictionary.npy") and os.path.exists("customer_dictionary.npy") and os.path.exists("order_dictionary.npy") and os.path.exists("order_week_dict.npy") and os.path.exists("price_data.pt"):
        print('Data already existst!:)')
        print('Loading orderline_data ...')
        orderline_data  = torch.load('orderline_data.pt')
        print('Loading price_data ...')
        price_data  = torch.load('price_data.pt')
        print('Loading article_dict ...')
        article_dict    = np.load('article_dictionary.npy',allow_pickle='TRUE').item()
        print('Loading customer_dict ...')
        customer_dict   = np.load('customer_dictionary.npy',allow_pickle='TRUE').item()
        print('Loading order_dict ...')
        order_dict      = np.load('order_dictionary.npy',allow_pickle='TRUE').item()
        print('Loading order_week_dict ...')
        order_week_dict      = np.load('order_week_dict.npy',allow_pickle='TRUE').item()
        return customer_dict, order_dict, article_dict, order_week_dict, orderline_data, price_data
    else:
        print('Cannot find data, data has to be transformed ...')
        orderlines = pd.read_csv('ORDERLINES.csv')
        prices = pd.read_csv('WEEKLY_PRICES.csv')  
        
        key_article_list = list(orderlines.KEY_ARTICLE.unique())
        key_customer_list = list(orderlines.KEY_CUSTOMER.unique())
        key_order_list    = list(orderlines.KEY_ORDER.unique())  
        key_week_list    = list(orderlines.KEY_WEEK.unique())
        key_week_list.sort()
        
        crop_frame = orderlines[['KEY_CUSTOMER','ORDER_NUM','KEY_WEEK']].drop_duplicates()
        keys = zip(crop_frame.KEY_CUSTOMER, crop_frame.ORDER_NUM)
        order_week_dict = dict(zip(keys,crop_frame.KEY_WEEK))
        
        customer_dict = {}
        for i in range(len(key_customer_list)):
            customer_dict[i] = key_customer_list[i] 
            customer_dict[key_customer_list[i]] = i
            
        article_dict = {}
        for i in range(len(key_article_list)):
            article_dict[i] = key_article_list[i]
            article_dict[key_article_list[i]] = i
                        
        order_dict = {}
        for i in range(len(key_order_list)):
            order_dict[i] = key_order_list[i]
            order_dict[key_order_list[i]] = i 
            
        week_dict = {}
        for i in range(len(key_week_list)):
            week_dict[i] = key_week_list[i]
            week_dict[key_week_list[i]] = i
        
        order_week_dict = {(customer_dict[k[0]],k[1]):week_dict[order_week_dict[k]] for k in list(order_week_dict.keys())}
        
        article_list  = list(article_dict[i] for i in key_article_list)
        customer_list = list(customer_dict[i] for i in key_customer_list)
        order_list  = list(order_dict[i] for i in key_order_list)
        week_list    = list(orderlines.KEY_WEEK.unique())
        
        # Data dimensions 
        no_customers = len(key_customer_list)
        orders_per_customer = max(orderlines['ORDER_NUM'])+1
        no_article = len(key_article_list)
        orderline_data = torch.zeros(no_customers,orders_per_customer,no_article)
        progress = ProgressBar(len(key_order_list) , fmt=ProgressBar.FULL)
        print('Transforming orders to tensor data ...') 
        
        
        last_order_dict = {} # deze misschien mee geven...
        
        for c in customer_list:
            cust_tens = torch.zeros(orders_per_customer,no_article)
            olCust = orderlines[orderlines['KEY_CUSTOMER']== customer_dict[c]]
            last_order_dict[c] = max(olCust.ORDER_NUM.unique()) 
            for o in olCust.ORDER_NUM.unique():
                articles_in_order = list(article_dict[a] for a in olCust[olCust['ORDER_NUM'] == o]['KEY_ARTICLE'])
                order_tens = order_to_tensor(articles_in_order,no_article)
                cust_tens[o,:] = order_tens
                progress.current  += 1
                progress()
            orderline_data[c] = cust_tens 
        progress.done()
        
        
        no_weeks = len(key_week_list)
        price_data = torch.zeros(no_weeks,2,no_article)
        progress = ProgressBar(len(week_list)*no_article , fmt=ProgressBar.FULL)
        print('Transforming prices to tensor data ...')  
        for week in key_week_list:
            week_prices = prices[prices["KEY_WEEK"] == week]
            week_tens = torch.zeros(2,no_article)
            for article in article_list:
                article_prices = week_prices[week_prices["KEY_ARTICLE"] == article_dict[article]]
                if article_prices.empty:
                    continue
                week_tens[0][article] = article_prices.PRICE_EU.item()
                week_tens[1][article] = article_prices.DISCOUNT.item()
                progress.current  += 1
                progress()
            price_data[week_dict[week]] = week_tens
        progress.done()
        
        
        print('Saving data for later use ...')
        torch.save(orderline_data, 'orderline_data.pt')
        torch.save(price_data, 'price_data.pt')
        np.save('article_dictionary.npy', article_dict)
        np.save('customer_dictionary.npy', customer_dict)
        np.save('order_dictionary.npy', order_dict)
        np.save('order_week_dict.npy', order_week_dict)
    
        return customer_dict, order_dict, article_dict, order_week_dict, orderline_data, price_data 

#%%
def train_val_test_split(full_data_tensor,split_vec = [0.6,0.2,0.2],random = False):
    rdn.seed(413)
    customers = list(range(full_data_tensor.size()[0]))
    if random:
        rdn.shuffle(customers)
        
    train_customers = customers[0:math.ceil(split_vec[0]*len(customers))]
    val_customers   = customers[math.ceil(split_vec[0]*len(customers)):math.ceil((split_vec[0]+split_vec[1])*len(customers))]
    test_customers  = customers[math.ceil((split_vec[0]+split_vec[1])*len(customers)):]
    
    train_data = full_data_tensor[train_customers]
    val_data = full_data_tensor[val_customers]
    test_data = full_data_tensor[test_customers]
    return train_data, train_customers, val_data, val_customers, test_data, test_customers 


def batch_generator(data, batch_size, random = False):
    no_batches = data.size()[0]/batch_size
    counter = 0 
    index = list(range(data.size()[0]))
    if random:
        rdn.shuffle(index)
    while 1:
        index_batch  = index[batch_size*counter:batch_size*(counter+1)]
        input_batch  = data[index_batch,:-1,:]
        target_batch  = data[index_batch,1:,:]
        seq_lengths = torch.sum(input_batch.abs().sum(dim=2).bool(), dim = 1, dtype = torch.long) -1
        for cust in range(input_batch.size()[0]):
            input_batch[cust,seq_lengths[cust].item(),:] = torch.zeros(input_batch.size()[2]) 
        counter += 1
        yield(index_batch, input_batch,target_batch,seq_lengths)
        if (counter >= no_batches):
            if random:
                np.random.shuffle(index)
            counter=0


def batch_generator_E1(data, price_data, customer_list, order_week_dict, batch_size, random = False):
    no_batches = data.size()[0]/batch_size
    counter = 0 
    index = list(range(data.size()[0]))
    if random:
        rdn.shuffle(index)
    while 1:
        index_batch   = index[batch_size*counter:batch_size*(counter+1)]
        customers = [customer_list[i] for i in index_batch]
        input_batch   = data[index_batch,:-1,:]
        target_batch  = data[index_batch,1:,:]
        seq_lengths = torch.sum(input_batch.abs().sum(dim=2).bool(), dim = 1, dtype = torch.long) -1
        for cust in range(input_batch.size()[0]):
            input_batch[cust,seq_lengths[cust].item(),:] = torch.zeros(input_batch.size()[2]) 
        output_prices = torch.zeros(target_batch.size()[0],target_batch.size()[1],2,target_batch.size()[2])
        for i in range(target_batch.size()[0]):
            customer = customers[i]
            seq_length = seq_lengths[i]
            for order in range(target_batch.size()[1]):
                if order == seq_length:
                    break
                order_num = order+1
                week_index = order_week_dict[(customer,order_num)]
                output_prices[i][order] = price_data[week_index]
        counter += 1
        yield(customers, input_batch,target_batch,seq_lengths, output_prices)
        if (counter >= no_batches):
            if random:
                np.random.shuffle(index)
            counter=0
            

def batch_generator_E1_1(data,base_predictions, price_data, customer_list, order_week_dict, batch_size, random = False):
    no_batches = data.size()[0]/batch_size
    counter = 0 
    index = list(range(data.size()[0]))
    if random:
        rdn.shuffle(index)
    while 1:
        index_batch   = index[batch_size*counter:batch_size*(counter+1)]
        customers = [customer_list[i] for i in index_batch]
        input_batch   = data[index_batch,:-1,:]
        seq_lengths = torch.sum(input_batch.abs().sum(dim=2).bool(), dim = 1, dtype = torch.long) -1
        target_batch  = data[index_batch,1:,:]
        input_batch = base_predictions[index_batch,:-1,:]
        for cust in range(input_batch.size()[0]):
            input_batch[cust,seq_lengths[cust].item():,:] = torch.zeros(input_batch.size()[1]-seq_lengths[cust].item(),input_batch.size()[2]) 
        output_prices = torch.zeros(target_batch.size()[0],target_batch.size()[1],2,target_batch.size()[2])
        for i in range(target_batch.size()[0]):
            customer = customers[i]
            seq_length = seq_lengths[i]
            for order in range(target_batch.size()[1]):
                if order == seq_length:
                    break
                order_num = order+1
                week_index = order_week_dict[(customer,order_num)]
                output_prices[i][order] = price_data[week_index]
        counter += 1
        yield(customers, input_batch,target_batch,seq_lengths, output_prices)
        if (counter >= no_batches):
            if random:
                np.random.shuffle(index)
            counter=0            
            
def batch_generator_E2(data, price_data, customer_list, order_week_dict, batch_size, random = False, return_week_mask = False):
    no_batches = data.size()[0]/batch_size
    counter = 0 
    index = list(range(data.size()[0]))
    if random:
        rdn.shuffle(index)
    while 1:
        index_batch   = index[batch_size*counter:batch_size*(counter+1)]
        customers = [customer_list[i] for i in index_batch]
        input_batch   = data[index_batch,:-1,:]
        target_batch  = data[index_batch,1:,:]
        seq_lengths = torch.sum(input_batch.abs().sum(dim=2).bool(), dim = 1, dtype = torch.long) -1
        for cust in range(input_batch.size()[0]):
            input_batch[cust,seq_lengths[cust].item(),:] = torch.zeros(input_batch.size()[2]) 
        output_prices = torch.zeros(target_batch.size()[0],target_batch.size()[1],2,target_batch.size()[2])
        input_prices  = torch.zeros(input_batch.size()[0],input_batch.size()[1],2,input_batch.size()[2])
        if return_week_mask != False:
            week_mask =  torch.zeros(batch_size,data.size()[1]-1)
        for i in range(target_batch.size()[0]):
            customer = customers[i]
            seq_length = seq_lengths[i]
            for order in range(target_batch.size()[1]):
                if order == seq_length:
                    break
                output_order_num = order+1
                input_order_num  = order
                
                output_week_index = order_week_dict[(customer,output_order_num)]
                input_week_index  = order_week_dict[(customer,input_order_num)]
                
                if return_week_mask != False:
                    if output_week_index in return_week_mask:
                        week_mask[i,order+1] = output_week_index
                        
                    
                    

                
                
                
                output_prices[i][order] = price_data[output_week_index]
                input_prices[i][order] = price_data[input_week_index]

        counter += 1
        if return_week_mask != False:
            yield(customers, input_batch,target_batch,seq_lengths, output_prices, input_prices, week_mask)
        else: 
            yield(customers, input_batch,target_batch,seq_lengths, output_prices, input_prices)
        if (counter >= no_batches):
            if random:
                np.random.shuffle(index)
            counter=0





