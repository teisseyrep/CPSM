import numpy as np
from sklearn.linear_model import LogisticRegression
from DNN.dnn import dnn

def bbsc(X_train,Z_train,Y_train,X_test,Z_test,clf_name=''):

    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)    
    n_train = Xall_train.shape[0]
    
    if clf_name=='':
        model = LogisticRegression(penalty=None)
        model.fit(Xall_train,Y_train)
        Y_pred_train = model.predict_proba(Xall_train)[:,1]
        Y_pred_test = model.predict_proba(Xall_test)[:,1]
    elif clf_name=='dnn':
        #p_1_xall= dnn(Xall_train, Y_train, Xall_test, weights0=[0])[:,1]
        model = dnn()
        model.fit(Xall_train, Y_train, weights0=[0])
        Y_pred_train = model.predict_proba(Xall_train)[:,1]
        Y_pred_test = model.predict_proba(Xall_test)[:,1]
    else:
        raise Warning('clf_name is incorrect!')    
    
    
    #model = clf
    #model.fit(Xall_train,Y_train)
    #Y_pred_train = model.predict(Xall_train)
    A = np.mean(Y_pred_train[np.where(Y_train==1)[0]])
    B = np.mean(Y_pred_train[np.where(Y_train==0)[0]])
    
    #Y_pred_test = model.predict(Xall_test)
    Q = np.mean(Y_pred_test)    
    eps = 0.0001
    q1 = (Q-B)/(A-B+eps)
    q0 = 1-q1
    p1 = np.mean(Y_train)
    p0 = 1-p1
    if p1==0:
        p1 = eps 
    if p0==0:
        p0 = eps
    
    w1 = q1/p1
    w0 = q0/p0

    weights = np.zeros(n_train)
    for i in np.arange(n_train):
         if Y_train[i]==1:
             weights[i] = w1  
         else:
             weights[i] = w0
     
    #Train weighted model:
    #model_w = LogisticRegression(penalty=None) 
    #model_w.fit(Xall_train,Y_train,sample_weight=weights)
     
    #P(y=1|x,z):
    #q_xall = model_w.predict_proba(Xall_test)

    if clf_name=='':
        model_w = LogisticRegression(penalty=None)
        model_w.fit(Xall_train,Y_train,sample_weight=weights)
        q_xall = model_w.predict_proba(Xall_test)
    elif clf_name=='dnn':
        #p_1_xall= dnn(Xall_train, Y_train, Xall_test, weights0=[0])[:,1]
        model_w = dnn()
        model_w.fit(Xall_train, Y_train, weights0=weights)
        q_xall = model_w.predict_proba(Xall_test)
    else:
        raise Warning('clf_name is incorrect!')   


    return q_xall
