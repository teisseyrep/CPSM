# CPSM

This is the repository for the paper 
>Paweł Teisseyre, Jan Mielniczuk, Probabilistic classification when conditional distribution of labels between source and target domains is shifted.

## Abstract ##

In many practical applications of machine learning, a discrepancy often arises between a source distribution from which labeled training examples are drawn and a target distribution for which only unlabeled data is observed. Traditionally, two main scenarios have been considered to address this issue: covariate shift (CS), where only the marginal distribution of features changes, and label shift (LS), which involves a change in the class variable's prior distribution. However, these frameworks do not encompass all forms of distributional shift. Recently, a novel assumption known as Sparse Joint Shift (SJS) has been proposed, generalizing both CS and LS by allowing changes in both class variable and certain features. This paper introduces a new setting, Conditional Probability Shift (CPS), which captures the most interesting case of SJS, namely when the conditional distribution of the class variable given some specific  features changes while the distribution of remaining features given the specific features and the class is preserved. For this scenario we present the Conditional Probability Shift Model (CPSM) based on modelling the class variable's conditional probabilities using multinomial regression.
Since the class variable is not observed for the target data, the parameters of the multinomial model for its distribution are estimated using the Expectation-Maximization algorithm. The proposed method is generic and  can be combined with any probabilistic classifier.
The effectiveness of CPSM is demonstrated through experiments on synthetic datasets and a case study using the MIMIC medical database, revealing its superior balanced classification accuracy on the target data compared to existing methods tailored either to LS or SJS scenarios, particularly in situations where the conditional distribution shifts without changes in prior probabilities, which are not detectable by LS-based methods.

## Data ##

The artificial datasets can be generated as described in the paper, using the function ``generate_artificial_data`` in the file ``artificial.py``. The user can change various parameters, such as number of observations, number of features x, number of features z, class priors p(y=1), q(y=1) and parameter k.
The function generates both source (train) and target (test) datasets.

Due to licensing reasons we cannot release the MIMIC dataset and therefore we only provide the code to run experiments on artificial data.

## Experiments ##

The file ``experiment_artificial.py`` contains the code to run experiment for artificial dataset. 
It is possible to choose a base classifier (logistic regression or neural network) and set various parameters needed to generate the artificial dataset. The scripts outputs balanced accuracy and approximation error calculated on the target (test) data, for the proposed CPSM metod as well as for the NAIVE method in which a classification model is trained on the source data and applied to the target data without any corrections. If you use DNN classifier, you can change the network architecture in the DNN module.

## Basic Structure of the module ##
1. ``CPSM.cpsm`` module contains implementation of the proposed method.
2.  ``DNN.dnn`` module  contains implementation of the neural network used in the experiments.
3.  ``NAIVE.naive`` module  contains implementation of the naive method.



## Contact

If you have any questions or issues, please reach out via my email:

> teisseyrep AT ipipan DOT waw DOT pl

