import numpy as np
from utils import sigma

def generate_artificial_data1(n,p,pz,pi_p=0.05,pi_q=0.5,k=1):

    """
    Generate artificial dataset
    
    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of features in X
    pz : int
        number of features in Z
    pi_p : int, optional
        P(Y=1)    
    pi_q : int, optional
        Q(Y=1)
    k : int, optional
        Q(Y=1|Z) = sigma(b0 + b^{T} X ), where b=(k,...,k) and b0 is chosen to control pi_q
        
    """
    

    # Generate Z:
    Z_train = np.zeros((n,pz))
    Z_test = np.zeros((n,pz))
    for j in np.arange(pz):
         Z_train[:,j] = np.random.binomial(1, 0.5, size=n)    
         Z_test[:,j] = np.random.binomial(1, 0.5, size=n)    
    
    # Generate Y:
    Y_train = np.zeros(n)
    Y_test = np.zeros(n)
       
    
    a0_vec = np.arange(-10,10,0.05)
    prior_diff = np.zeros(a0_vec.shape[0])
    i=0
    for a0 in a0_vec:
        a = np.zeros(pz)
        probs_train = sigma(a0 + np.dot(Z_train,a))
        Y_train = np.random.binomial(1, probs_train, size=n)
        prior_diff[i] = np.abs(pi_p-np.mean(Y_train))
        i=i+1
    sel =np.argmin(prior_diff)    
    a0 = a0_vec[sel]
    probs_train = sigma(a0 + np.dot(Z_train,a))
    Y_train = np.random.binomial(1, probs_train, size=n)  
    
    b = np.repeat(k,pz)
    b0_vec = np.arange(-10,10,0.05)
    prior_diff = np.zeros(b0_vec.shape[0])
    i = 0
    for b0 in b0_vec:
        probs_test = sigma(b0 + np.dot(Z_test,b))
        Y_test = np.random.binomial(1, probs_test, size=n)    
        prior_diff[i] = np.abs(pi_q-np.mean(Y_test))
        i=i+1
        
    sel =np.argmin(prior_diff)    
    b0 = b0_vec[sel]
    probs_test = sigma(b0 + np.dot(Z_test,b))
    Y_test = np.random.binomial(1, probs_test, size=n)    
        
    
    X_train = np.zeros((n,p))
    X_test = np.zeros((n,p))        
    
    if p<=pz+1:
        raise Warning('It must be p>pz+1')
    
    Sigma = np.diag(np.ones(p))
    for i in np.arange(n):
          mu = np.concatenate(([2*Y_train[i]],0.2*Z_train[i,:],np.zeros(p-pz-1)),axis=0)
          X_train[i,:] = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1)
          mu = np.concatenate(([2*Y_test[i]],0.2*Z_test[i,:],np.zeros(p-pz-1)),axis=0)
          X_test[i,:] = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1)
    
    
    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)

    return Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test


def generate_artificial_data2(n,p,pz,pi_p=0.05,pi_q=0.5,k=1):

    """
    Generate artificial dataset
    
    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of features in X
    pz : int
        number of features in Z
    pi_p : int, optional
        P(Y=1)    
    pi_q : int, optional
        Q(Y=1)
    k : int, optional
        Q(Y=1|Z) = sigma(b0 + b^{T} X ), where b=(k,...,k) and b0 is chosen to control pi_q
        
    """
    

    # Generate Z:
    Z_train = np.zeros((n,pz))
    Z_test = np.zeros((n,pz))
    for j in np.arange(pz):
         Z_train[:,j] = np.random.normal(0, 1, size=n)    
         Z_test[:,j] = np.random.normal(0, 1, size=n)    
    
    # Generate Y:
    Y_train = np.zeros(n)
    Y_test = np.zeros(n)
       
    
    a0_vec = np.arange(-10,10,0.05)
    prior_diff = np.zeros(a0_vec.shape[0])
    i=0
    for a0 in a0_vec:
        a = np.zeros(pz)
        probs_train = sigma(a0 + np.dot(Z_train,a))
        Y_train = np.random.binomial(1, probs_train, size=n)
        prior_diff[i] = np.abs(pi_p-np.mean(Y_train))
        i=i+1
    sel =np.argmin(prior_diff)    
    a0 = a0_vec[sel]
    probs_train = sigma(a0 + np.dot(Z_train,a))
    Y_train = np.random.binomial(1, probs_train, size=n)  
    
    b = np.repeat(k,pz)
    b0_vec = np.arange(-10,10,0.05)
    prior_diff = np.zeros(b0_vec.shape[0])
    i = 0
    for b0 in b0_vec:
        probs_test = sigma(b0 + np.dot(Z_test,b))
        Y_test = np.random.binomial(1, probs_test, size=n)    
        prior_diff[i] = np.abs(pi_q-np.mean(Y_test))
        i=i+1
        
    sel =np.argmin(prior_diff)    
    b0 = b0_vec[sel]
    probs_test = sigma(b0 + np.dot(Z_test,b))
    Y_test = np.random.binomial(1, probs_test, size=n)    
        
    
    X_train = np.zeros((n,p))
    X_test = np.zeros((n,p))        
    
    if p<=pz+1:
        raise Warning('It must be p>pz+1')
    
    Sigma = np.diag(np.ones(p))
    for i in np.arange(n):
          mu = np.concatenate(([2*Y_train[i]],0.2*Z_train[i,:],np.zeros(p-pz-1)),axis=0)
          X_train[i,:] = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1)
          mu = np.concatenate(([2*Y_test[i]],0.2*Z_test[i,:],np.zeros(p-pz-1)),axis=0)
          X_test[i,:] = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1)
    
    
    Xall_train = np.concatenate( (X_train, Z_train),axis=1)
    Xall_test = np.concatenate( (X_test, Z_test),axis=1)

    return Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test
