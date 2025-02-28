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

#Generate artificial data:
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





















