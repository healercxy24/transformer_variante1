# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from data_process import *
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)


#%% hyperparameters and datas

seq_len = 40
dataset_name = 'FD001'


#train_data, test_data, truth_label = dataset_process(dataset_name)
dataset = get_dataset(dataset_name, seq_len);
train_seq = dataset['lower_train_seq_tensor'] # size [16000, 50, 18] [dataset_len, seq_len, num_features]
#train_seq = train_seq.view(train_seq.shape[0], -1) # [dataset_len, seq_len*num_features] [16000, 1300]
train_label = dataset['lower_train_label_tensor'] # [16000] [dataset_len]

valid_seq = dataset['lower_valid_seq_tensor']
#valid_seq = valid_seq.view(valid_seq.shape[0], -1)
valid_label = dataset['lower_valid_label_tensor']   # [4581]

test_seq = dataset['lower_test_seq_tensor']   # size [13046, 50, 18]
test_label = dataset['lower_test_label_tensor']


d_model = seq_len


#%% training

def train(model, criterion, optimizer, batch_size):
    model.train()  # turn on train mode

    total_train_loss = 0
    num_batches = train_seq.shape[0] // batch_size
    src_mask = generate_square_subsequent_mask(train_seq.shape[2]).float().to(device)
    
    for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):

        # compute the loss for the lower-level
        inputs, targets = get_batch(train_seq, train_label, i, batch_size) #[40, 256, 18] [256, 40]
        inputs = inputs.permute(2, 1, 0).float()   # [18, 256, 40]
        targets = targets.reshape(1, batch_size, seq_len).float() # [1, 256, 40]
        predictions = model(inputs, src_mask)   #[50,50,1]
        #print(predictions)
        loss = criterion(predictions, targets)        
            
        optimizer.zero_grad()      
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        
        total_train_loss += loss.item()
        
    total_train_loss /= num_batches
    
    return total_train_loss


def evaluate(model, criterion, batch_size):

    model.eval()  # turn on evaluation mode

    total_valid_loss = 0
    num_batches = valid_seq.shape[0] // batch_size
    src_mask = generate_square_subsequent_mask(train_seq.shape[2]).to(device)
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):

            # compute the loss for the lower-level
            inputs, targets = get_batch(train_seq, train_label, i, batch_size) #[40, 256, 18] [256, 40]
            inputs = inputs.permute(2, 1, 0).float()   # [18, 256, 40]
            targets = targets.reshape(1, batch_size, seq_len).float() # [1, 256, 40]
            predictions = model(inputs, src_mask)   #[50,50,1]
            #print(predictions)
            loss = criterion(predictions, targets)               
            
            total_valid_loss += loss.item()
            
        total_valid_loss = total_valid_loss / num_batches
        

    return total_valid_loss


def test(model, criterion, batch_size):

    model.eval()  # turn on evaluation mode

    total_test_loss = 0
    num_batches = test_seq.shape[0] // batch_size
    src_mask = generate_square_subsequent_mask(test_seq.shape[2]).to(device)
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, num_batches*batch_size, batch_size)):
            # compute the loss for the lower-level
            inputs, targets = get_batch(train_seq, train_label, i, batch_size) #[40, 256, 18] [256, 40]
            inputs = inputs.permute(2, 1, 0).float()   # [18, 256, 40]
            targets = targets.reshape(1, batch_size, seq_len).float() # [1, 256, 40]
            predictions = model(inputs, src_mask)   #[50,50,1]
            #print(predictions)
            loss = criterion(predictions, targets)               
            
            total_test_loss += loss.item()
            
        total_test_loss = total_test_loss / num_batches
        

    return total_test_loss
    
    
#%% running

import time
import optuna
import plotly
import operator

def objective(trial):
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-1, 10.0) 
    #learning_rate = 2.0
    nlayers = trial.suggest_int('nlayers', 2, 6)
    dropout = trial.suggest_loguniform('dropout', 0.001, 0.5)
    nhid = trial.suggest_int('nhid', 50, 600, 50)    
    nhead = trial.suggest_int('nhead', 8, 8)
    #nhead = 2
    batch_size = trial.suggest_int('batch_size', 256, 256)
    #seq_len = trial.suggest_int('seq_len', 20, 100, 5)
    num_epochs = 10
    
    
    #model = torch.load('temp_model_FD001_18.pk1').to(device)
    model = Transformer(d_model, nhead, nhid, nlayers, dropout).to(device)     
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    #best_result = study.best_value
    best_result = float('inf')
    
    
    
    trainloss = []
    validloss = []
    
    for epoch in range(1, num_epochs + 1):
        
        epoch_start_time = time.time()

        train_loss = train(model, criterion, optimizer, batch_size)
        valid_loss = evaluate(model, criterion, batch_size)
        test_loss = test(model, criterion, batch_size)
        
        trainloss.append(train_loss)
        validloss.append(valid_loss)
        
        scheduler.step()

        
        if epoch % 1 == 0:
            
            elapsed = time.time() - epoch_start_time
            
            print('-' * 89)

            print(f'| end of epoch: {epoch:3d} | time: {elapsed:5.2f}s | ')
            print(f' | train loss: {train_loss:5.2f} ')
            print(f' | valid loss: {valid_loss:5.2f} ')
            print(f' | test loss: {test_loss:5.2f} ')
            #print(optimizer.state_dict()['param_groups'][0]['lr'])

            print('-' * 89)
            
            
            # save the best result with the smallest test loss
            store_addr = 'temp_model_' + dataset_name + "_" + str(d_model) + '.pk1' 
            if test_loss < best_result:
                best_result = test_loss            
                torch.save(model, store_addr)
    
    print(f' | test loss: {test_loss:5.2f} ')
    torch.save(model, 'temp_model.pk1')
    
    # plot
    plt.plot(range(num_epochs), trainloss, label='train loss')
    plt.plot(range(num_epochs), validloss, label='valid loss')
    plt.legend()
    plt.show()

    return best_result


study_store_addr_li = "sqlite:///%s_fea%s_li.db" % (dataset_name, str(d_model))
study_store_addr_HI = "sqlite:///%s_fea%s_HI.db" % (dataset_name, str(d_model))
#study = optuna.create_study(study_name='linearpredict_optim', direction="minimize", storage = study_store_addr_li, load_if_exists=True)
study = optuna.create_study(study_name='HIpredict_optim_'+ dataset_name, direction="minimize", storage = study_store_addr_HI, load_if_exists=True)  
study.optimize(objective, n_trials=1)


print('study.best_params:', study.best_params)
print('study.best_value:', study.best_value)

"""
with pca: M = 4
study.best_params: {'dropout': 0.16599790295000072, 'nhid': 250, 'nlayers': 3}
study.best_value: 1047.9029934359532

with pca: M = 8
study.best_params: {'dropout': 0.20687966329224702, 'nhid': 400, 'nlayers': 8}
study.best_value: 1380.0305190740846
"""