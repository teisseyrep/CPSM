import numpy as np
from tqdm import tqdm
from logistic import logistic_optimize
from sklearn.linear_model import LogisticRegression
from utils import sigma
from DNN.dnn import dnn

def cpsm(X_train,Z_train,Y_train,X_test,Z_test,clf_name='',epochs = 50,calibrate=False):

    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)    

    if clf_name=='':
        model = LogisticRegression(penalty=None)
        model.fit(Xall_train,Y_train)
        p_1_xall = model.predict_proba(Xall_test)[:,1]
    elif clf_name=='dnn':
        #p_1_xall= dnn(Xall_train, Y_train, Xall_test, weights0=[0])[:,1]
        deepCLF = dnn()
        deepCLF.fit(Xall_train, Y_train, weights0=[0])
        p_1_xall = deepCLF.predict_proba(Xall_test)[:,1]
    else:
        raise Warning('clf_name is incorrect!')    


    #model = clf
    #model.fit(Xall_train,Y_train)
    #p_1_xall = model.predict_proba(Xall_test)[:,1]
    
    ### Calibration: ###
    if calibrate==True: 
        #p_1_xall_train= dnn(Xall_train, Y_train, Xall_train, weights0=[0])[:,1]
        p_1_xall_train = deepCLF.predict_proba(Xall_train)[:,1]
        clf_cal = LogisticRegression(penalty=None)
        clf_cal.fit(p_1_xall_train.reshape(-1,1),Y_train)
        p_1_xall = sigma(clf_cal.intercept_ + clf_cal.coef_*p_1_xall)[0,:]
    
    modelZ =  LogisticRegression(penalty=None) 
    modelZ.fit(Z_train,Y_train)
    p_1_z = modelZ.predict_proba(Z_test)[:,1]
    
    q_1_z = np.repeat(0.5,Z_test.shape[0])
    
    for epoch in tqdm(range(epochs)):
    
        # E-step:
        if epoch==0:
             q_1_xall = p_1_xall   
        else:    
            d1 = p_1_xall * (q_1_z/p_1_z)
            d2 = (1-p_1_xall) * ((1-q_1_z) / (1-p_1_z))
            q_1_xall = d1 / (d1+d2)
        
        # M-step:
        hat_b, risk, q_1_z = logistic_optimize(Z_test, q_1_xall,epochs=1000,lr=0.1)


    q_xall = np.zeros((X_test.shape[0],2))
    q_xall[:,0] = 1-q_1_xall
    q_xall[:,1] = q_1_xall

    return q_xall








