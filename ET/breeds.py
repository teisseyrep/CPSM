#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
path = '/path/to/directory/'
sys.path.append(path + 'data/breeds/')
sys.path.append(path + 'utils/')
from data import BREEDS
from transfer import exp_tilt, fit_source_classifier, get_tilt, model_with_target_samples
from transfer import get_source_weights 
from transfer import eval_source_target, weighted_training
import numpy as np
import torch, itertools, json


def breeds(mix_prop = 0.01, subsample = 6000, epochs = 1000, lr = 1e-3,
           batch_size = 1500, reg_normalizer = 1e-4, ITER = 0, calibrate = 'None',
           device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    """
    mix_prop: mix propoportion for target data
    subsample: sample sizes for source and target, fix -1 for running the experiment on full data
    epochs: epochs
    lr: learning rate
    batch_size: minibatch size
    reg_normalizer: regularizer for normalizer
    ITER: setting seed
    calibrate: calibration for source classifier
    device: cpu or cuda
    
    """
    
    np.random.seed(ITER+12345)
    torch.manual_seed(ITER+12345)
    
    data = BREEDS(mix_prop=mix_prop, subsample = subsample)
    Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y = data
    ss, st, tt, _, _ = eval_source_target(data, fit_prop = 1)
    try:
        acc_target_samples = model_with_target_samples(data)
    except:
        acc_target_samples = -1
    
    
    
    source_classifier = fit_source_classifier(Str_x, Str_y, Sts_x, Sts_y, fit_prop = 1, calibrate=calibrate)
    
    
    # defining sufficient statistics T(x)
    # here we use T(x) = x
    def suff_stat(x):
        return x
    
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
    target_classifier = weighted_training(Str_x, Str_y, weights, Tts_x, Tts_y)
    tilt_score = target_classifier.score(Tts_x, Tts_y)
    
    return ss, st, tt, tilt_score, KL, acc_target_samples


v_mix_prop = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04,  0.08, 0.16, ]
reg_tilt = 1e-6
epochs = 500
lr = 1e-4
batch_size = 1500
reg_normalizer = 1e-6
subsample = 600
calibrate = 'BCTS'
v_ITER = list(range(10))

grid = itertools.product(v_mix_prop, v_ITER)
grid = list(grid)

job = int(float(sys.argv[1]))
mix_prop, ITER = grid[job]


result = breeds(mix_prop=mix_prop, subsample = subsample,
                reg_tilt=reg_tilt, epochs = epochs, 
                lr = lr, batch_size = batch_size,
                reg_normalizer=reg_normalizer, ITER = ITER,
                calibrate = calibrate)


ss, st, tt, tilt_score, KL, acc_target_samples, ws = result

result_dict = {'mix_prop': mix_prop, 'subsample': subsample,
               'reg_tilt': reg_tilt, 'epochs': epochs, 
               'lr': lr, 'batch_size': batch_size,
               'reg_normalizer': reg_normalizer, 'iter': ITER, 'ss': ss, 
               'st': st, 'tt': tt, 'acc_target_samples': acc_target_samples,
               }
result_dict['tilt'] = float(tilt_score)
result_dict['KL'] = float(KL)

with open(path + f'breeds/summary/summary_{job}.json', 'w') as f:
    json.dump(result_dict, f)
    