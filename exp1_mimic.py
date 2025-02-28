import numpy as np
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
import pandas as pd
import time

from DNN.dnn import dnn
from CPSEM.cpsem import cpsem
from EM.em import em
from BBSC.bbsc import bbsc
from SEES.sees import sees_custom
from ET.et import et_custom
from NAIVE.naive import naive

from sampling import  sample_fixed_cond_probs
from utils import mi_filter



clf_name='' #logistic
#clf_name = 'dnn'

#zvar = '' #Age
zvar = 'sex'

k_seq = np.array([0.3,0.5,0.7])
n_sym = 5

a = 0.05
#a = 0.1
#a = 0.2
#a = 0.5
b = 0.1



for target in np.array(['copd','diabetes','kidney','fluid']):
    print('\n ======== TARGET: ' + target + ' ======== \n')    
 
    method_seq = np.array(['cpsem','naive','bbsc','em','sees','et'])
    
    
    for method in method_seq:
        print('\n === Method: ' + method + ' === \n')
        
        res_bacc = np.zeros((n_sym,len(k_seq)))
        res_bacc1 = np.zeros((n_sym,len(k_seq)))
        res_acc = np.zeros((n_sym,len(k_seq)))
        res_error = np.zeros((n_sym,len(k_seq)))
        res_time = np.zeros((n_sym,len(k_seq)))
        
        counter = 0
        for k in k_seq: 
            print('\n --- k= ' + str(k) + ' --- \n')
            
            for sym in np.arange(n_sym):
                print("|",end='')
    
    
                #Read and prepare MIMIC dataset:
                d01 = arff.loadarff(open('../MIMIC2/holdout-1/training_part1.arff', 'rt'))
                d02 = arff.loadarff(open('../MIMIC2/holdout-1/testval_part2.arff', 'rt'))
                d03 = arff.loadarff(open('../MIMIC2/holdout-1/testval_part3.arff', 'rt'))
                df1 = pd.DataFrame(d01[0])
                df2 = pd.DataFrame(d02[0])
                df3 = pd.DataFrame(d03[0])
                frames = [df1, df2, df3]
                df0 = pd.concat(frames)
                colnames = df0.columns
                colnames = colnames.to_numpy()
                xnames = colnames[0:310]
                ynames = colnames[310:320]
                df0X = df0[xnames]
                df0Y = df0[ynames]
                Xall = df0X.to_numpy(dtype='float32')
                Ydiseases = df0Y.to_numpy()
                if target=='copd':
                    target_index = 0
                elif target=='diabetes':
                    target_index = 1
                elif target=='fluid':
                    target_index = 2
                elif target=='hypertension':
                    target_index = 3
                elif target=='hypotension':
                    target_index = 4
                elif target=='kidney':
                    target_index = 5
                elif target=='lipoid':
                    target_index = 6
                elif target=='liver':
                    target_index = 7
                elif target=='thrombosis':
                    target_index = 8
                elif target=='thyroid':
                    target_index = 9
                    
                Xall[:,128] = Xall[:,128] * 22.35 + 67.48 #Age variable (back to original scale)
                Yall = np.where(Ydiseases[:,target_index]==b'1',1,0)
                    
                
                samp = sample_without_replacement(Xall.shape[0], 5000, random_state=sym)
                Xall = Xall[samp,:]
                Yall = Yall[samp]
                
                #TRAIN-TEST SPLIT:
                Xall_train, Xall_test, Y_train, Y_test = train_test_split(Xall, Yall, test_size=0.5, random_state=sym)
                
                if zvar=='':
                    Z_train = np.zeros((Xall_train.shape[0],1))
                    Z_train[:,0] = np.where(Xall_train[:,128]>60,1,0)
                    p = Xall_train.shape[1]
                    sdiff = np.setdiff1d(np.arange(p), np.array([128]))
                    X_train = Xall_train[:,sdiff]
                    Z_test = np.zeros((Xall_test.shape[0],1))
                    Z_test[:,0] = np.where(Xall_test[:,128]>60,1,0)
                    X_test = Xall_test[:,sdiff]
                elif zvar=='sex':
                    Z_train = np.zeros((Xall_train.shape[0],1))
                    Z_train[:,0] = np.where(Xall_train[:,129]>0,1,0)
                    p = Xall_train.shape[1]
                    sdiff = np.setdiff1d(np.arange(p), np.array([129]))
                    X_train = Xall_train[:,sdiff]
                    Z_test = np.zeros((Xall_test.shape[0],1))
                    Z_test[:,0] = np.where(Xall_test[:,129]>0,1,0)
                    X_test = Xall_test[:,sdiff]                    
                
                sel = mi_filter(X_train,Y_train,pmax=30)
                X_train = X_train[:,sel]
                X_test = X_test[:,sel]
                
                
                X_train,Z_train,Y_train = sample_fixed_cond_probs(X_test,Z_test,Y_test,p1_z0 = a,p1_z1 = a)
                X_test,Z_test,Y_test = sample_fixed_cond_probs(X_test,Z_test,Y_test,p1_z0 = b,p1_z1 = b+k)
                
                
                Xall_train = np.concatenate( (X_train, Z_train),axis=1)
                Xall_test = np.concatenate( (X_test, Z_test),axis=1)
                
                
                start_time = time.time()
                #ORACLE METHOD:
                if clf_name=='':
                    modelT = LogisticRegression(penalty=None)
                    modelT.fit(Xall_test,Y_test)
                    q_xall_oracle = modelT.predict_proba(Xall_test)
                elif clf_name=='dnn':
                    #p_1_xall= dnn(Xall_train, Y_train, Xall_test, weights0=[0])[:,1]
                    modelT = dnn()
                    modelT.fit(Xall_test, Y_test, weights0=[0])
                    q_xall_oracle = modelT.predict_proba(Xall_test)
                else:
                    raise Warning('clf_name is incorrect!')    
    
                #NAIVE METHOD:
                if method=='naive':    
                    q_xall = naive(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)
                #CPSEM method:            
                elif method=='cpsem':    
                    q_xall = cpsem(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name,epochs = 500)
                #EM method:
                elif method == 'em':    
                    q_xall = em(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name,epochs = 500)    
                #BBSC method:
                elif method=='bbsc':    
                    q_xall = bbsc(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)    
                #SEES method:
                elif method=='sees':    
                    q_xall = sees_custom(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)
                #ET method:
                elif method=='et':    
                    q_xall = et_custom(X_train,Z_train,Y_train,X_test,Z_test,Y_test,clf_name=clf_name,epochs=500) 
                
                end_time = time.time()
                run_time = end_time - start_time
                
                res_bacc[sym,counter] = balanced_accuracy_score(Y_test, np.where(q_xall[:,1]>0.5,1,0))
                res_bacc1[sym,counter] = balanced_accuracy_score(Y_test, np.where(q_xall[:,1]>np.mean(q_xall[:,1]),1,0))
                res_acc[sym,counter] = accuracy_score(Y_test, np.where(q_xall[:,1]>0.5,1,0))
                res_error[sym,counter] = np.mean(np.abs(q_xall[:,1]-q_xall_oracle[:,1]))
                res_time[sym,counter] = run_time
    
            counter = counter + 1
                    
                
        np.savetxt('results_mimic/exp1_bacc_' + method  + '_' + target + '_a_' + str(a)  + '_b_' + str(b)  + clf_name + zvar + '.txt', res_bacc, fmt='%1.3f')
        np.savetxt('results_mimic/exp1_bacc1_' + method  + '_' + target + '_a_' + str(a)  + '_b_' + str(b)  + clf_name + zvar + '.txt', res_bacc1, fmt='%1.3f')
        np.savetxt('results_mimic/exp1_acc_' + method  + '_' + target + '_a_' + str(a)  + '_b_' + str(b)  + clf_name + zvar + '.txt', res_acc, fmt='%1.3f')
        np.savetxt('results_mimic/exp1_error_' + method  + '_' + target + '_a_' + str(a)  + '_b_' + str(b)  + clf_name + zvar + '.txt', res_error, fmt='%1.3f')
        np.savetxt('results_mimic/exp1_time_' + method  + '_' + target + '_a_' + str(a)  + '_b_' + str(b)  + clf_name + zvar + '.txt', res_time, fmt='%1.3f')
                
            
            
            
          


