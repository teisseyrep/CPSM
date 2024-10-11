from sklearn.feature_selection import mutual_info_classif
import numpy as np    
from sklearn.preprocessing import KBinsDiscretizer

def sigma(x):
    res = np.exp(x)/(1+np.exp(x))
    return res

def softmax(a):
    res = np.exp(a)/np.sum(np.exp(a))
    return res

def mi_filter(X,y,pmax=50):
    mi = np.zeros(X.shape[1])
    for j in np.arange(X.shape[1]):    
        mi[j] = mutual_info_classif(X[:,j].reshape(-1,1), y)
    sel = np.argsort(-mi)[0:pmax]
    return sel

