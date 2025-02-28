# CPSM

This is the repository for the paper 
>Paweł Teisseyre, Jan Mielniczuk, A generalized approach to label shift: the Conditional Probability Shift Model.

## Abstract ##

In many practical applications of machine learning, a discrepancy often arises between a source distribution from which labeled training examples are drawn and a target distribution for which only unlabeled data is observed. Traditionally, two main scenarios have been considered to address this issue: covariate shift (CS), where only the marginal distribution of features changes, and label shift (LS), which involves a change in the class variable's prior distribution. However, these frameworks do not encompass all forms of distributional shift. Recently, a novel assumption known as Sparse Joint Shift (SJS) has been proposed, generalizing both CS and LS by allowing changes in both class variable and certain features. This paper introduces a new setting, Conditional Probability Shift (CPS), which captures the most interesting case of SJS, namely when the conditional distribution of the class variable given some specific  features changes while the distribution of remaining features given the specific features and the class is preserved. For this scenario we present the Conditional Probability Shift Model (CPSM) based on modelling the class variable's conditional probabilities using multinomial regression.
Since the class variable is not observed for the target data, the parameters of the multinomial model for its distribution are estimated using the Expectation-Maximization algorithm. The proposed method is generic and  can be combined with any probabilistic classifier.
The effectiveness of CPSM is demonstrated through experiments on synthetic datasets and a case study using the MIMIC medical database, revealing its superior balanced classification accuracy on the target data compared to existing methods tailored either to LS or SJS scenarios, particularly in situations where the conditional distribution shifts without changes in prior probabilities, which are not detectable by LS-based methods.

## Data ##

The artificial datasets can be generated as described in the paper, using the function ``generate_artificial_data1`` and ``generate_artificial_data2`` in the file ``artificial.py``. The user can change various parameters, such as number of observations, number of features x, number of features z, class priors p(y=1), q(y=1) and parameter k.
The function generates both source (train) and target (test) datasets.

Due to licensing reasons we cannot release the MIMIC dataset and therefore we only provide the code to run experiments on artificial data.

## Experiments ##

The file ``experiment_artificial.py`` contains the code to run experiment for artificial dataset. 
It is possible to choose a base classifier (logistic regression or neural network) and set various parameters needed to generate the artificial dataset. The scripts outputs balanced accuracy and approximation error calculated on the target (test) data, for the proposed CPSM metod as well as for the NAIVE method in which a classification model is trained on the source data and applied to the target data without any corrections. If you use DNN classifier, you can change the network architecture in the DNN module.

## Basic Structure of the module ##
1. ``CPSM.cpsm`` module contains implementation of the proposed method.
2.  ``DNN.dnn`` module  contains implementation of the neural network used in the experiments.
3.  ``NAIVE.naive`` module  contains implementation of the naive method.


Examples
--------
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


from DNN.dnn import dnn
from CPSM.cpsm import cpsm
from NAIVE.naive import naive
from EM.em import em
from BBSC.bbsc import bbsc
from SEES.sees import sees_custom
from ET.et import et_custom

from artificial import generate_artificial_data1

# Parameters:
n = 1000 #size of source/target data
p = 10 #number of features x
pz = 5 #number of features z
pi_p = 0.05 # source class prior p(y=1)
pi_q = 0.05 # target class prior q(y=1)
k = 5

# Base classifier:
clf_name='' #logistic
#clf_name = 'dnn'

#Genertate artificial data:
Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test = generate_artificial_data1(n,p,pz,pi_p=pi_p,pi_q=pi_q,k=k)
           
#ORACLE METHOD:
if clf_name=='':
    modelT = LogisticRegression(penalty=None)
    modelT.fit(Xall_test,Y_test)
    q_xall_oracle = modelT.predict_proba(Xall_test)
elif clf_name=='dnn':
    modelT = dnn()
    modelT.fit(Xall_test, Y_test, weights0=[0])
    q_xall_oracle = modelT.predict_proba(Xall_test)
else:
    raise Warning('clf_name is incorrect!')    

#NAIVE METHOD:
q_xall_naive = naive(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)
print('Balanced accuracy for method NAIVE=',balanced_accuracy_score(Y_test, np.where(q_xall_naive[:,1]>0.5,1,0)))

#CPSM method:            
q_xall_cpsm = cpsm(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name,epochs = 500)
print('Balanced accuracy for method CPSM=',balanced_accuracy_score(Y_test, np.where(q_xall_cpsm[:,1]>0.5,1,0)))


#MLLS method:
q_xall_em = em(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name,epochs = 500)    
print('Balanced accuracy for method MLLS=',balanced_accuracy_score(Y_test, np.where(q_xall_em[:,1]>0.5,1,0)))

#BBSC method:
q_xall_bbsc = bbsc(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)    
print('Balanced accuracy for method BBSC=',balanced_accuracy_score(Y_test, np.where(q_xall_bbsc[:,1]>0.5,1,0)))
    
#SEES method:
q_xall_sees = sees_custom(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name)
print('Balanced accuracy for method SEES=',balanced_accuracy_score(Y_test, np.where(q_xall_sees[:,1]>0.5,1,0)))

#ET method:
q_xall_et = et_custom(X_train,Z_train,Y_train,X_test,Z_test,Y_test,clf_name=clf_name,epochs=500) 
print('Balanced accuracy for method ET=',balanced_accuracy_score(Y_test, np.where(q_xall_et[:,1]>0.5,1,0)))
```


## Contact

If you have any questions or issues, please reach out via my email:

> teisseyrep AT ipipan DOT waw DOT pl

