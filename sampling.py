import numpy as np


def sample_fixed_class_prior(X_train,Z_train,Y_train,p1=0.1):

    """
    Downsampling: the function samples a subset of the give dataset  such that P(Y=1)=p1 in the output data.

    """    

    if p1<=np.mean(Y_train):
        w0 = np.where(Y_train==0)[0]
        n0  = len(w0)
        sel_pos = int((p1/(1-p1)) * n0)
        w1 = np.where(Y_train==1)[0]
        w1sel = np.random.choice(w1, size=sel_pos, replace=False)
        sel = np.concatenate((w0, w1sel),0)
        X_train_samp = X_train[sel,:]
        Z_train_samp = Z_train[sel,:]
        Y_train_samp= Y_train[sel]
    else:
        w1 = np.where(Y_train==1)[0]
        n1  = len(w1)
        sel_neg = int(((1-p1)/p1) * n1)
        w0 = np.where(Y_train==0)[0]
        w0sel = np.random.choice(w0, size=sel_neg, replace=False)
        sel = np.concatenate((w0sel, w1),0)
        X_train_samp = X_train[sel,:]
        Z_train_samp = Z_train[sel,:]
        Y_train_samp= Y_train[sel]
    
    return X_train_samp, Z_train_samp, Y_train_samp


def sample_fixed_cond_probs(X,Z,Y,p1_z0 = 0.1,p1_z1 = 0.7):

    """
    Downsampling on layers Z=0 and Z=1: the function samples a subset of the give dataset  such that P(Y=1|Z=0)=p1_z0 and P(Y=1|Z=1)=p1_z1 in the output data.

    """    

    wz1 = np.where(Z==1)[0]
    wz0 = np.where(Z==0)[0]
    
    X0 = X[wz0,:]
    Z0 = Z[wz0,:]
    Y0 =  Y[wz0]
    X1 = X[wz1,:]
    Z1 = Z[wz1,:]
    Y1 =  Y[wz1]
    
    X1_samp, Z1_samp, Y1_samp = sample_fixed_class_prior(X1,Z1,Y1,p1=p1_z1)
    X0_samp, Z0_samp, Y0_samp = sample_fixed_class_prior(X0,Z0,Y0,p1=p1_z0)
    
    X_samp = np.concatenate((X0_samp,X1_samp),0)
    Z_samp = np.concatenate((Z0_samp,Z1_samp),0)
    Y_samp = np.concatenate((Y0_samp,Y1_samp),0)

    return X_samp,Z_samp,Y_samp



