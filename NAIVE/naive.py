import numpy as np
from sklearn.linear_model import LogisticRegression
from DNN.dnn import dnn

def naive(X_train,Z_train,Y_train,X_test,Z_test,clf_name):

    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)     

    if clf_name=='':
        model = LogisticRegression(penalty=None)
        model.fit(Xall_train,Y_train)
        q_xall = model.predict_proba(Xall_test)
    elif clf_name=='dnn':
        model = dnn()
        model.fit(Xall_train, Y_train, weights0=[0])
        q_xall = model.predict_proba(Xall_test)
        #q_xall = dnn(Xall_train, Y_train, Xall_test, weights0=[0])
    else:
        raise Warning('clf_name is incorrect!')

    
    return q_xall
