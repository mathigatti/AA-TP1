# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree, svm
from sklearn.metrics import roc_auc_score, recall_score, precision_score,fbeta_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus as pydot
from collections import Counter,defaultdict
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mail_attributes import *
from  aa_tp_utils import *
from  sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

train_data_frame = open_set('jsons/mail_training_set.json','train_set')

process_attributes(train_data_frame)
train_data_frame.X = train_data_frame[train_data_frame.attributes].values
train_data_frame.y = train_data_frame['class']
yBool = booleanizar(train_data_frame.y)


test_data_frame = open_set('jsons/mail_testing_set.json','test_set')
process_attributes(test_data_frame)
test_data_frame.X = test_data_frame[test_data_frame.attributes].values
test_data_frame.y = test_data_frame['class']
test_data_frame.yBool = booleanizar(test_data_frame.y)

param_grid = {'alpha':[0.0,0.5,1.0],
		'fit_prior' : [ True, False],
		}
f05_score = make_scorer(fbeta_score, beta=0.5)

clf = GaussianNB()
grid_search = GridSearchCV(clf,param_grid={}, scoring=f05_score,n_jobs=3,cv=5,verbose=1)
grid_search.fit(train_data_frame.X, yBool)


csv_file = open('./plots/cv_grid_search_nb_Gaussian' + 'f05' + '.csv', "w")
csv_file.write('mean_score,std dev score,alpha,fit_prior\n')
for test in grid_search.grid_scores_:
	csv_file.write(str(np.mean(test[2]))+','+str(np.std(test[2])))
	for value in test[0].itervalues():
		csv_file.write(',' + str(value))
	csv_file.write('\n')

csv_file.close()

print grid_search.best_estimator_
print grid_search.best_score_

print 'Testing performance in Testing Data Gaussian'

y_pred = grid_search.predict(test_data_frame.X)
print 'fbeta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

clf = BernoulliNB()
grid_search = GridSearchCV(clf,param_grid=param_grid, scoring=f05_score,n_jobs=3,cv=5,verbose=1)
grid_search.fit(train_data_frame.X, yBool)


csv_file = open('./plots/cv_grid_search_nb_Bernoulli' + 'f05' + '.csv', "w")
csv_file.write('mean_score,std dev score,alpha,fit_prior\n')
for test in grid_search.grid_scores_:
	csv_file.write(str(np.mean(test[2]))+','+str(np.std(test[2])))
	for value in test[0].itervalues():
		csv_file.write(',' + str(value))
	csv_file.write('\n')

csv_file.close()

print grid_search.best_estimator_
print grid_search.best_score_

print 'Testing performance in Testing Data Bernoulli'

y_pred = grid_search.predict(test_data_frame.X)
print 'fbeta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)

clf = MultinomialNB()
grid_search = GridSearchCV(clf,param_grid=param_grid, scoring=f05_score,n_jobs=3,cv=5,verbose=1)
grid_search.fit(train_data_frame.X, yBool)


csv_file = open('./plots/cv_grid_search_nb_Multinomial' + 'f05' + '.csv', "w")
csv_file.write('mean_score,std dev score,alpha,fit_prior\n')
for test in grid_search.grid_scores_:
	csv_file.write(str(np.mean(test[2]))+','+str(np.std(test[2])))
	for value in test[0].itervalues():
		csv_file.write(',' + str(value))
	csv_file.write('\n')

csv_file.close()

print grid_search.best_estimator_
print grid_search.best_score_

print 'Testing performance in Testing Data Multinomial'

y_pred = grid_search.predict(test_data_frame.X)
print 'fbeta 0.5 on testing = ',fbeta_score(test_data_frame.yBool,y_pred,0.5)
