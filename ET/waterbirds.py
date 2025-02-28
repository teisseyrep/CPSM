#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys, json
path = '/path/to/the/project/'
sys.path.append(path + 'data/waterbirds/')
sys.path.append(path + 'utils/')
from waterbirds_LR import load_waterbirds_data_full
from transfer import exp_tilt, fit_source_classifier, get_tilt
from transfer import get_source_weights
from transfer import weighted_training
import torch, itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from time import gmtime, strftime



def waterbirds(full_data, calibrate = 'None', target_env = [0, 1, 2, 3],
               ITER = 0, lr = 4e-5, batch_size = 200,
               reg_normalizer = 0, epochs = 500, 
               device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    """
    Getting waterbirds results
    
    full_data: training and test data of the form (train_x, train_y, train_g), (test_x, test_y, test_g)
    calibrate: source model calibrateion
    target_env: groups in target domain 
    ITER: iteration
    lr: learning rate for ExTRA
    batch_size: batch size for ExTRA
    reg_normalizer: regularizer for ExTRA normalizer
    epochs: epochs for ExTRA
    
    """
    
    (train_x, train_y, train_g), (test_x, test_y, test_g) = full_data    
    
    # setting seeds
    np.random.seed(ITER)
    torch.manual_seed(ITER)
        
    
    # saving parameters in return dictionary
    return_dict = dict(epochs = epochs, lr = lr,
                       batch_size = batch_size, reg_normalizer = reg_normalizer,
                       ITER = int(ITER), calibrate = calibrate, 
                       )
    
    # target environment from test data
    conditions = np.zeros(shape = (test_g.shape[0], len(target_env)), dtype = bool)
    for i, e in enumerate(target_env):
        conditions[:, i] = test_g == e
    condition = conditions.any(axis = 1)
    target_x, target_y, target_g = test_x[condition], test_y[condition], test_g[condition]
    
    target_samples = np.unique(target_g, return_counts=True)
    ts = dict()
    for i in range(target_samples[0].shape[0]):
        ts[int(target_samples[0][i])] = int(target_samples[1][i])
    
    return_dict['target-env'] = ts
    
    
    
    # train test split
    Str_x, Sts_x, Str_y, Sts_y, Str_g, Sts_g = train_test_split(train_x, train_y, train_g, random_state=0, test_size=0.1)
    Ttr_x, Tts_x, Ttr_y, Tts_y, Ttr_g, Tts_g = train_test_split(target_x, target_y, target_g, random_state=0, test_size=0.1)
    Str_it, Sts_it = np.zeros_like(Str_y, dtype = np.bool), np.zeros_like(Str_y, dtype = np.bool)
    data = Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y
    
    
    # training of weighted logistic regression to balance groups    
    weight_group = 1/np.unique(Str_g, return_counts=True)[1]
    weights_gr = np.zeros_like(Str_g)
    for i, w in enumerate(weight_group):
        weights_gr = weights_gr +  (Str_g == i).astype('float') * w
        
    weighted_lr = LogisticRegression().fit(Str_x, Str_y, sample_weight=weights_gr)
    return_dict['weighted-acccurcay'] = float(weighted_lr.score(Tts_x, Tts_y))  
    
    # source classifier
    source_classifier = fit_source_classifier(Str_x, Str_y, Sts_x, Sts_y, fit_prop = 1, calibrate=calibrate)
    return_dict['st'] = float(np.mean((source_classifier(Tts_x)[:, 1] >0.5).astype('float') == Tts_y))
    
    
    # defining sufficient statistics T(x)
    # here we use T(x) = x
    def suff_stat(x):
        return x
        
    ################################
    ##########   ExTRA   ###########    
    tilt = get_tilt(suff_stat(Str_x), Str_y, regularizer=0)
    tilt, KL  = exp_tilt(data = data, source_classifier = source_classifier,
                        tilt = tilt, suff_stat= suff_stat, 
                        epochs = epochs,
                        lr = lr, batch_size=batch_size,
                        reg_normalizer=reg_normalizer, device = device)
    cpu_device = torch.device('cpu')
    tilt = tilt.to(cpu_device)
    tilt.A = tilt.A.to(cpu_device)
    weights = get_source_weights(Str_x, Str_y, tilt, suff_stat)
    
    ## fitting target classifier with ExTRA weights
    target_classifier = weighted_training(Str_x, Str_y, weights, Tts_x, Tts_y)
    return_dict['tilt'] = float(target_classifier.score(Tts_x, Tts_y))
    return_dict['KL'] = float(KL)
    full_weights = get_source_weights(train_x, train_y, tilt, suff_stat)
    return_dict['source-weights'] = [float(w) for w in full_weights]
    
    
    # target environment from test data
    conditions = np.zeros(shape = (Str_g.shape[0], len(target_env)), dtype = bool)
    for i, e in enumerate(target_env):
        conditions[:, i] = Str_g == e
    condition = conditions.any(axis = 1)
    target_x, target_y, target_g = Str_x[condition], Str_y[condition], Str_g[condition]
    lr_tt = LogisticRegression(solver = 'lbfgs', C = 0.1, tol = 1e-6, max_iter=500).fit(target_x, target_y)
    return_dict['tt'] = lr_tt.score(Tts_x, Tts_y)
    
    
    
    return return_dict

  

if __name__ == '__main__':
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ## parameter grid
    calibrates =    ['None', 'TS', 'BCTS', 'VS']
    target_envs = [[0, 2], [1, 2], [0, 3], [1, 3], [0, 1, 2, 3]]
    ITERS = range(10)
    lrs =  [5e-4, 4e-5,]
    batch_sizes = [500, 1000,]
    epochss = [100, 200, 400, ]
    rns = [1e-6, 0]
    # njobs = 4 * 5 * 10 * 1 * 2 * 2 * 3 * 2 = 4800
    
    grid = list(itertools.product(calibrates, target_envs, ITERS, lrs, batch_sizes, epochss, rns))
    
    ## job number
    i = int(float(sys.argv[1]))
    
    ## loading waterbirds data
    full_data, n_groups = load_waterbirds_data_full(path + 'data/waterbirds/', label_noise=0., frac=1.0)

    ## waterbirds results
    calibrate, target_env, ITER, lr, batch_size, epochs, rn = grid[i]
    r = waterbirds(full_data = full_data, calibarte = calibrate, target_env = target_env,
            ITER = ITER, lr = lr, batch_size = batch_size,
            epochs = epochs, reg_normalizer = rn)
    
    t = strftime("%a,%d-%b-%Y,%H:%M:%S", gmtime())
    with open(path + f'waterbirds/summary/summary_{i}_time:'+t+'.json', 'w+') as f:
        json.dump(r, f)




