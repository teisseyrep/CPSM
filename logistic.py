import numpy as np
from utils import sigma, softmax

def logistic_grad(X1,probs,Y,lam,theta):
    n = X1.shape[0]
    resid = Y-probs
    resid = resid.reshape(-1,1)
    res = -np.dot(Y-probs,X1)/n + 2*lam*theta
    return res


def logistic_optimize(X,Y,lr=0.01,epochs=1000,lam=0):

    n = X.shape[0]  
    p = X.shape[1]
    Xconst = np.repeat(1,n).reshape(-1,1)
    X1 = np.concatenate((Xconst, X),1)
    
    risk = []
    theta = np.repeat(0,p+1)
    
    for epoch in np.arange(epochs):
        eta = np.dot(X1,theta)
        probs = sigma(eta)

        risk_current = -np.mean(Y * np.log(probs)  + (1-Y) * np.log(1-probs)) + lam * sum(theta[1:]**2)
        risk.append(risk_current)
        grad = logistic_grad(X1,probs,Y,lam,theta)
        theta = theta -lr * grad
        
    return theta, risk, probs 



def multinomial_grad(X1,probs,Ym,lam,theta):
    n = X1.shape[0]
    p = X1.shape[1]
    K = Ym.shape[1]
    res = np.zeros((p,K))
    
    for k in np.arange(K):
        res[:,k] = -np.dot(Ym[:,k]-probs[:,k],X1)/n + 2*lam*theta[:,k]
    
    return res



def multinomial_optimize(X,Ym,lr=0.01,epochs=1000,lam=0):

    K = Ym.shape[1]
    n = X.shape[0]  
    p = X.shape[1]
    Xconst = np.repeat(1,n).reshape(-1,1)
    X1 = np.concatenate((Xconst, X),1)
    
    risk = []
    theta = np.zeros((p+1,K))
    
    for epoch in np.arange(epochs):
        
        probs = np.zeros((n,K))
        eta = np.zeros((n,K))
        for k in np.arange(K):
            eta[:,k] = np.exp( np.dot(X1,theta[:,k]) )
        eta_sum = np.sum(eta,1)
        for k in np.arange(K):
            probs[:,k] = eta[:,k] /eta_sum
        

    
        risk_current = -np.mean( np.sum(Ym*np.log(probs),1) )
        risk.append(risk_current)
        grad = multinomial_grad(X1,probs,Ym,lam,theta)
        for k in np.arange(K):
            theta[:,k] = theta[:,k] -lr * grad[:,k]
     
        theta[:,0] = 0

    return theta, risk, probs



