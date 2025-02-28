import numpy as np
import copy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from DNN.dnn import dnn

def em(X_train,Z_train,Y_train,X_test,Z_test,clf_name='',epochs = 200):

    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)    
    
    # model = clf
    # model.fit(Xall_train,Y_train)
    # p_1_xall = model.predict_proba(Xall_test)[:,1]
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
    
    p_1 = np.mean(Y_train)
    q_1 = 0.5
    
    for epoch in tqdm(range(epochs)):
    
        # E-step:
        if epoch==0:
             q_1_xall = p_1_xall   
        else:    
            d1 = p_1_xall * (q_1/p_1)
            d2 = (1-p_1_xall) * ((1-q_1) / (1-p_1))
            q_1_xall = d1 / (d1+d2) #P(y=1|x,z)
        
        # M-step:
        q_1 = np.mean(q_1_xall)


    q_xall = np.zeros((X_test.shape[0],2))
    q_xall[:,0] = 1-q_1_xall
    q_xall[:,1] = q_1_xall

    return q_xall


# def em_multi(X_train,Z_train,Y_train,X_test,Z_test,clf,epochs = 200):

#     K = len(np.unique(Y_train))
#     Xall_train = np.concatenate( (X_train, Z_train),axis=1)
#     Xall_test = np.concatenate( (X_test, Z_test),axis=1)    
    
#     model = clf
#     model.fit(Xall_train,Y_train)
#     p_xall = model.predict_proba(Xall_test)
    
#     p = np.zeros(K)
#     for k in np.arange(K):
#         p[k] = len(np.where(Y_train==k)[0])/Y_train.shape[0]
#     q = np.repeat(0.5,K)
    
#     for epoch in tqdm(range(epochs)):
    
        
#         # E-step:
#         if epoch==0:
#              q_xall = copy.deepcopy(p_xall)
#         else:    
#             d1 = p_xall * (q/p)
#             d2 = np.sum(d1,1)
#             for k in np.arange(K):
#                  q_xall[:,k] = d1[:,k]/d2
        
#         # M-step:
#         q = np.mean(q_xall,0)


#     return q_xall


