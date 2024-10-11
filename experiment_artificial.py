import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


from DNN.dnn import dnn
from CPSM.cpsm import cpsm
from NAIVE.naive import naive
from artificial import generate_artificial_data

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
Y_train, Y_test, X_train, X_test, Z_train, Z_test, Xall_train, Xall_test = generate_artificial_data(n,p,pz,pi_p=pi_p,pi_q=pi_q,k=k)
           
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

#CPSM method:            
q_xall_cpsm = cpsm(X_train,Z_train,Y_train,X_test,Z_test,clf_name=clf_name,epochs = 500)


print('Approximation error for method Naive=',np.mean(np.abs(q_xall_naive[:,1]-q_xall_oracle[:,1])))
print('Approximation error for method CPSM=',np.mean(np.abs(q_xall_cpsm[:,1]-q_xall_oracle[:,1])))


print('Balanced accuracy for method NAIVE=',balanced_accuracy_score(Y_test, np.where(q_xall_naive[:,1]>0.5,1,0)))
print('Balanced accuracy for method CPSM=',balanced_accuracy_score(Y_test, np.where(q_xall_cpsm[:,1]>0.5,1,0)))

















