import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import time

from DNN.dnn import dnn
from CPSEM.cpsem import cpsem
from EM.em import em
from BBSC.bbsc import bbsc
from SEES.sees import sees_custom
from ET.et import et_custom
from NAIVE.naive import naive

from artificial import generate_artificial_data1, generate_artificial_data2

# Parameters:
ds = 'ds1'
n = 1000
p = 10
pz = 5
pi_p = 0.05

pi_q = 0.05
#pi_q = 0.5
#pi_q = 0.8

clf_name='' #logistic
#clf_name = 'dnn'


k_seq = np.array([0,5])
n_sym = 30

method_seq = np.array(['naive','bbsc','em','sees','cpsem','et'])

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

            if ds=='ds1':
                Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test = generate_artificial_data1(n,p,pz,pi_p=pi_p,pi_q=pi_q,k=k)
            if ds=='ds2':
                Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test = generate_artificial_data2(n,p,pz,pi_p=pi_p,pi_q=pi_q,k=k)                
           
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
            
            res_bacc1[sym,counter] = balanced_accuracy_score(Y_test, np.where(q_xall[:,1]>np.mean(q_xall[:,1]),1,0))
            res_bacc[sym,counter] = balanced_accuracy_score(Y_test, np.where(q_xall[:,1]>0.5,1,0))
            res_acc[sym,counter] = accuracy_score(Y_test, np.where(q_xall[:,1]>0.5,1,0))
            res_error[sym,counter] = np.mean(np.abs(q_xall[:,1]-q_xall_oracle[:,1]))
            res_time[sym,counter] = run_time

        counter = counter + 1

    np.savetxt('results_artificial/exp1_acc_' + method  + '_' + ds + '_n' + str(n) + '_p' + str(p) + '_pz' + str(pz) + '_pip' + str(pi_p) + '_piq' + str(pi_q) +  '.txt', res_acc, fmt='%1.3f')
    np.savetxt('results_artificial/exp1_bacc_' + method  + '_' + ds + '_n' + str(n) + '_p' + str(p) + '_pz' + str(pz) + '_pip' + str(pi_p) + '_piq' + str(pi_q) +  '.txt', res_bacc, fmt='%1.3f')
    np.savetxt('results_artificial/exp1_bacc1_' + method  + '_' + ds + '_n' + str(n) + '_p' + str(p) + '_pz' + str(pz) + '_pip' + str(pi_p) + '_piq' + str(pi_q) +  '.txt', res_bacc1, fmt='%1.3f')
    np.savetxt('results_artificial/exp1_error_' + method  + '_' + ds + '_n' + str(n) + '_p' + str(p) + '_pz' + str(pz) + '_pip' + str(pi_p) + '_piq' + str(pi_q) +  '.txt', res_error, fmt='%1.3f')
    np.savetxt('results_artificial/exp1_time_' + method  + '_' + ds + '_n' + str(n) + '_p' + str(p) + '_pz' + str(pz) + '_pip' + str(pi_p) + '_piq' + str(pi_q) +  '.txt', res_time, fmt='%1.3f')



 


















