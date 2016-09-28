# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree, svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score, make_scorer
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from IPython.display import Image 
#import pydotplus as pydot
from mail_attributes import *
from mail_utils import *
from collections import Counter,defaultdict
from sklearn import preprocessing
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from os.path import exists,isfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from aa_tp_utils import *

train_data_frame = open_set('jsons/mail_training_set.json','train_set')
process_attributes(train_data_frame)
train_data_frame.X  = train_data_frame[train_data_frame.attributes].values
train_data_frame.y = train_data_frame['class']
train_data_frame.yBool = booleanizar(train_data_frame.y)

test_data_frame = open_set('jsons/mail_testing_set.json','test_set')
process_attributes(test_data_frame)
test_data_frame.X = test_data_frame[test_data_frame.attributes].values
test_data_frame.y = test_data_frame['class']
test_data_frame.yBool = booleanizar(test_data_frame.y)


# Elijo mi clasificador.
dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=14)
rfc = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=14,max_features='sqrt',n_estimators=35)
gnb = GaussianNB()
bnb = BernoulliNB(fit_prior=True,alpha=1.0)
mnb = MultinomialNB(fit_prior=True,alpha=1.0)
knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', leaf_size=20,algorithm='kd_tree', p=1)
f05_score = make_scorer(fbeta_score, beta=0.5)
# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
print 'Fitting Decision Tree'
dtc.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = dtc.predict(test_data_frame.X)
print 'Decision Tree f beta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

print 'Fitting Random Forest'
rfc.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = rfc.predict(test_data_frame.X)
print 'Random Forest fbeta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5) 

print 'Fitting Gaussian Naive Bayes'
gnb.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = gnb.predict(test_data_frame.X)
print 'Gaussian Naive Bayes f beta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

print 'Fitting Bernoulli Naive Bayes'
bnb.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = bnb.predict(test_data_frame.X)
print 'Bernoulli Naive Bayes f beta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

print 'Fitting Multinomial Naive Bayes'
mnb.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = mnb.predict(test_data_frame.X)
print 'Multinomial Naive Bayes f beta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

print 'Fitting K Nearest Neighbors'
knn.fit(train_data_frame.X,train_data_frame.yBool)
y_pred = knn.predict(test_data_frame.X)
print 'K Nearest Neighbors f beta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

