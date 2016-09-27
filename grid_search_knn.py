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
from sklearn.neighbors import KNeighborsClassifier

train_data_frame = open_set('jsons/mail_training_set.json','train_set')

process_attributes(train_data_frame)
train_data_frame.X = train_data_frame[train_data_frame.attributes].values
train_data_frame.y = train_data_frame['class']
train_data_frame.y_binarized = preprocessing.label_binarize(train_data_frame.y,classes=['ham','spam'])
train_data_frame.y_binarized = np.array([number[0] for number in train_data_frame.y_binarized])

clf = KNeighborsClassifier()
param_grid = {'n_neighbors':[1,2,4,6],
		'weights':['uniform	','distance']
		'algorithm' : [ 'ball_tree', 'kd_tree'],
		'leaf_size' : [20, 30, 40],
		'p' : [1,2]
		}
f05_score = make_scorer(fbeta_score, beta=0.5)
grid_search = GridSearchCV(clf,param_grid=param_grid, scoring=f05_score,n_jobs=4,cv=5)
grid_search.fit(train_data_frame.X, train_data_frame.y_binarized)


csv_file = open('./plots/cv_grid_search_tree' + 'f05' + '.csv', "w")
csv_file.write('mean_score,std dev score, n_neighbors,weights, p, leaf_size, algorithm\n')
for test in grid_search.grid_scores_:
	csv_file.write(str(np.mean(test[2]))+','+str(np.std(test[2])))
	for value in test[0].itervalues():
		csv_file.write(',' + str(value))
	csv_file.write('\n')

csv_file.close()

print grid_search.best_estimator_
print grid_search.best_score_

print 'Testing performance in Testing Data'

test_data_frame = open_set('jsons/mail_testing_set.json','test_set')
test_data_frame.X = test_data_frame[test_data_frame.attributes].values
test_data_frame.y = test_data_frame['class']
test_data_frame.y_binarized = preprocessing.label_binarize(test_data_frame.y,classes=['ham','spam'])
test_data_frame.y_binarized = np.array([number[0] for number in test_data_frame.y_binarized])
process_attributes(test_data_frame)

y_pred = grid_search.predict(test_data_frame.X)
print 'fbeta 0.5 on testing = ',fbeta_score(test_data_frame.y_binarized,y_pred,0.5)