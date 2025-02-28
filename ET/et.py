#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.linear_model import LogisticRegression
# torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from ET.calibration import TempScaling, VectorScaling
from sklearn.model_selection import train_test_split
from DNN.dnn import dnn

def suff_stat(x):
        return x

def et_custom(X_train,Z_train,Y_train,X_test,Z_test,Y_test,clf_name='',epochs=200):
    
    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)  

    Str_x, Sts_x, Str_y, Sts_y = train_test_split(Xall_train, Y_train, random_state=0, test_size=0.1)
    Ttr_x, Tts_x, Ttr_y, Tts_y = train_test_split(Xall_test, Y_test, random_state=0, test_size=0.1)
    Str_it, Sts_it = np.zeros_like(Str_y, dtype = bool), np.zeros_like(Str_y, dtype = bool)
    data = Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y


            
    source_classifier = fit_source_classifier(Str_x, Str_y, Sts_x, Sts_y, fit_prop = 1, calibrate='None')  
    tilt = get_tilt(suff_stat(Str_x), Str_y, regularizer=0)
    tilt, KL  = exp_tilt(data = data, source_classifier = source_classifier,
                        tilt = tilt, suff_stat= suff_stat, 
                        epochs = epochs,
                        lr = 1e-4, batch_size=200,
                        reg_normalizer=0, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    cpu_device = torch.device('cpu')
    tilt = tilt.to(cpu_device)
    tilt.A = tilt.A.to(cpu_device)
    weights = get_source_weights(Str_x, Str_y, tilt, suff_stat)    

    wna = np.where(np.isnan(weights))
    if len(wna)>0:
        weights[wna]=1

    #Train weighted model:
    #model_w = LogisticRegression(penalty=None) 
    #model_w.fit(Str_x,Str_y,sample_weight=weights) 
    #q_xall = model_w.predict_proba(Xall_test)
    if clf_name=='':
        model_w = LogisticRegression(penalty=None)
        model_w.fit(Str_x,Str_y,sample_weight=weights)
        q_xall = model_w.predict_proba(Xall_test)
    elif clf_name=='dnn':
        #p_1_xall= dnn(Xall_train, Y_train, Xall_test, weights0=[0])[:,1]
        model_w = dnn()
        model_w.fit(Str_x, Str_y, weights0=weights)
        q_xall = model_w.predict_proba(Xall_test)
    else:
        raise Warning('clf_name is incorrect!')   
    
    
    return q_xall


def fit_source_classifier(Str_x, Str_y, Sts_x, Sts_y, weights = None,
                          fit_prop = 1, calibrate = 'None', cv = 3):
    
    
    """
    Fits source classifier
    
    Str_x: source train x
    Str_y: source train y
    Sts_x: source test x
    Sts_y: source test y
    weights: sample weights for (Str_x, Str_y), uniform if None
    fit_prop: proportion of data to be fitted on
    calibrate: calibration techniques for source classifier
    cv: cross-validation number for model calibration
    """
    
    if isinstance(weights, type(None)):
        weights = np.ones_like(Str_y)
    
    print('Fitting on', str(100 * fit_prop) + '% data...')
    
    
    n_train = int(Str_y.shape[0] * fit_prop)
    x_train = Str_x[:n_train]
    y_train = Str_y[:n_train]
    weights_train = weights[:n_train]
    

    if calibrate == 'None':
        sc = LogisticRegression(solver = 'lbfgs', C = 0.1,
                                tol = 1e-6, max_iter=500).fit(x_train,
                                                              y_train,
                                                              sample_weight=weights_train)
    elif calibrate == 'sklearn':
        base_clf = LogisticRegression(solver = 'lbfgs',
                                      C = 0.1, tol = 1e-6, max_iter=500)
        sc = CalibratedClassifierCV(base_estimator=base_clf,
                                    cv=cv).fit(x_train, y_train, sample_weight=weights_train)
    
    
    elif calibrate == 'BCTS' or calibrate == 'VS' or calibrate == 'TS':
        
        sc = LogisticRegression(solver = 'lbfgs', C = 0.1,
                                tol = 1e-6, max_iter=500).fit(x_train, y_train, sample_weight=weights_train)
        
        if calibrate == 'BCTS':
            calibrator = TempScaling(verbose=False, bias_positions='all')
        elif calibrate == 'VS':
            calibrator = VectorScaling(verbose = False)
        elif calibrate == 'TS':
            calibrator = TempScaling(verbose= False)
            
    else:
        raise ValueError('Calibration not implemented yet.')
        
        
    accuracy_train = sc.score(Sts_x, Sts_y)
    print('Source on source accuracy', accuracy_train)
    
    def source_classifier(x):
        p = sc.predict_proba(x)
        if calibrate == 'None' or calibrate == 'sklearn':
            return p
        elif calibrate == 'BCTS' or calibrate == 'VS' or calibrate == 'TS':
            
            n_classes = np.unique(Sts_y).shape[0]
            Sts_y_onehot = to_categorical(Sts_y, num_classes=n_classes)
            pf = calibrator(sc.predict_proba(Sts_x), Sts_y_onehot, posterior_supplied = True)
            return pf(p)
    
    
    
    return source_classifier



def get_tilt(Str_x, Str_y, A = None, regularizer = 0.0):
    
    nS, d = Str_x.shape
    y_unique, class_count = np.unique(Str_y, return_counts = True)
    n_classes = y_unique.shape[0]
    
    
    if type(A) == type(None):
        A = torch.eye(d)
        
    _, ld = A.shape
    
    class TILT(torch.nn.Module):

        def __init__(self):
            super(TILT, self).__init__()
            self.A = A
            self.linear = torch.nn.Linear(ld, n_classes)
    
        def forward(self, x):
            x = torch.matmul(x, self.A)
            x = self.linear(x)
            return x
    
    tilt = TILT()
    return tilt


def exp_tilt(data, source_classifier, tilt, suff_stat, lr = 1e-4,
             epochs = 10, batch_size = 1500, reg_normalizer = 1e-2,
             device = torch.device('cpu'), gamma = 1, optimizer_type = 'Adam'):
    
    Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y = data
    
    Str_s, Ttr_s = suff_stat(Str_x), suff_stat(Ttr_x)
    
    nS, d = Str_x.shape
    nT, d = Ttr_x.shape
    y_unique, class_count = np.unique(Str_y, return_counts = True)
    n_classes = y_unique.shape[0]
    
    Str_y_onehot = to_categorical(Str_y, num_classes=n_classes)
    Ttr_y_onehot = to_categorical(Ttr_y, num_classes=n_classes)
    Ttr_p = source_classifier(Ttr_x)
    
    t_Str_x = torch.from_numpy(Str_x).type(torch.float32).to(device)
    t_Str_s = torch.from_numpy(Str_s).type(torch.float32).to(device)
    t_Str_y = torch.from_numpy(Str_y_onehot).type(torch.float32).to(device)
    t_Ttr_x = torch.from_numpy(Ttr_x).type(torch.float32).to(device)
    t_Ttr_s = torch.from_numpy(Ttr_s).type(torch.float32).to(device)
    t_Ttr_y = torch.from_numpy(Ttr_y_onehot).type(torch.float32).to(device)
    t_Ttr_p = torch.from_numpy(Ttr_p).type(torch.float32).to(device)
    tilt = tilt.to(device)
    tilt.A = tilt.A.to(device)
    
    ns, nt = Str_x.shape[0], Ttr_x.shape[0]
    n = np.min([ns, nt])
    t_Str_s, t_Str_y = t_Str_s[:n], t_Str_y[:n]
    t_Ttr_s, t_Ttr_y, t_Ttr_p = t_Ttr_s[:n], t_Ttr_y[:n], t_Ttr_p[:n]
    
    dataset = TensorDataset(t_Str_s, t_Str_y, t_Ttr_s, t_Ttr_y, t_Ttr_p)
    loader = DataLoader(dataset = dataset, batch_size = batch_size)
    
    if optimizer_type == 'Adam':
        optimizer = optim.Adam([{'params': tilt.linear.parameters() ,'lr': lr}])
    else:
        optimizer = optim.SGD([{'params': tilt.linear.parameters() ,'lr': lr}])
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}/{epochs}')
        with tqdm(loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                
                # --- recollect source_x, source_y and target_x from batch data
                
                x_source, y_source, x_target, _, p_target = batch_data 
                
                optimizer.zero_grad()
                log_tilt_source = tilt(x_source) 
                log_tilt_max = torch.max(log_tilt_source)
                log_tilt_source = log_tilt_source.clone() - log_tilt_max
                tilt_source = torch.exp(log_tilt_source) * y_source
                normalizer = torch.mean(tilt_source) * n_classes
                    
                # if np.isnan(normalizer.detach().numpy()):
                #     print('Normalizer is nan at step', step)
                #     densities = torch.sum(p_target * tilt_target, axis = 1).detach().numpy()
                #     print('-- with minimum density from previous step', densities.min())
                #     print('-- with KL from previous step', KL.detach().numpy())
                #     # return densities, 0
                #     break     
                
                tilt_target = torch.exp(tilt(x_target)- log_tilt_max) / (normalizer)
                KL = - torch.mean(torch.log(torch.sum(p_target * tilt_target, axis = 1)))
                loss = KL +  reg_normalizer * (1/normalizer  + normalizer) 
            
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(KL=KL.item(), normalizer=normalizer.item())
        
            
        if epoch % 100 == 0:        
            x_source, y_source, x_target, p_target = t_Str_s, t_Str_y, t_Ttr_s, t_Ttr_p
            log_tilt_source = tilt(x_source) * y_source
            log_tilt_max = torch.max(log_tilt_source)
            log_tilt_source = log_tilt_source.clone() - log_tilt_max
            tilt_source = torch.exp(log_tilt_source)
            normalizer = torch.mean(tilt_source) * n_classes
            tilt_target = torch.exp(tilt(x_target)- log_tilt_max) / (normalizer)
            KL = - torch.mean(torch.log(torch.sum(p_target * tilt_target, axis = 1)))
            print('After epoch', epoch, 'KL:', KL.cpu().detach().numpy())
        
        scheduler.step()
        
    
    return tilt.to(torch.device('cpu')), KL.cpu().detach().numpy()



def get_source_weights(Str_x, Str_y, tilt, suff_stat):
    
    device = torch.device('cpu')
    y_unique = np.unique(Str_y)
    n_classes = y_unique.shape[0]
    Str_y_onehot = to_categorical(Str_y, num_classes=n_classes)
    t_Str_x = torch.from_numpy(suff_stat(Str_x)).type(torch.float32).to(device)
    
    log_weights = tilt(t_Str_x).detach().numpy()
    log_weights = log_weights - log_weights.max()
    
    weights = np.exp(log_weights) * Str_y_onehot
    normalizer = weights.mean() * n_classes
    weights = np.sum(weights, axis = 1) / normalizer
    
    return weights

def model_with_target_samples(data):
    Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y = data
    lr = LogisticRegression(solver = 'lbfgs', C = 0.1, tol = 1e-6, max_iter=500).fit(Str_x[Str_it == 1], Str_y[Str_it == 1])
    accuracy = lr.score(Tts_x, Tts_y)    
    print('Accuracy of model fitted only on target', accuracy)    
    return accuracy
    

def get_means(x, y):
    
    n_classes = np.unique(y).shape[0]
    _, d = x.shape
    means = np.zeros(shape = (n_classes, d))
    for i, k in enumerate(np.unique(y)):
        xk = x[y == k]
        means[i, :] = xk.mean(axis = 0)
        
    return means

def shift_means(x, y, shift):
    
    n_classes = np.unique(y).shape[0]
    y_onehot = to_categorical(y, num_classes=n_classes)
    return x + y_onehot @ shift
    
def eval_source_target(data, fit_prop = 1):
    
    print('Fitting on', str(100 * fit_prop) + '% data...')
    
    Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y = data
    
    # source logistic
    n = int(Str_x.shape[0] * fit_prop)
    lr_S = LogisticRegression(solver = 'lbfgs', C = 0.1, tol = 1e-6, max_iter=500)
    lr_S.fit(Str_x[:n], Str_y[:n])

    # target logistic
    n = int(Ttr_x.shape[0] * fit_prop)
    lr_T = LogisticRegression(solver='lbfgs', C = 0.1, tol = 1e-6, max_iter=500)
    lr_T.fit(Ttr_x[:n], Ttr_y[:n])

    ss_score, st_score, tt_score = lr_S.score(Sts_x, Sts_y), lr_S.score(Tts_x, Tts_y), lr_T.score(Tts_x, Tts_y)

    print('Source accuracy for source model', ss_score)
    print('Target accuracy for source model', st_score)
    print('Target accuracy for target model', tt_score)   
    
    return ss_score, st_score, tt_score, lr_S, lr_T


def weighted_training(Str_x, Str_y, weights, Tts_x, Tts_y, ):
    classifier_target = LogisticRegression(solver = 'lbfgs', C = 0.1, tol = 1e-6, max_iter=500).fit(Str_x, Str_y, weights)
    test_accuracy = classifier_target.score(Tts_x, Tts_y)
    print('Test accuracy', test_accuracy)
    return classifier_target
    
