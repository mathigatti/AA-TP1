# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree, svm
from sklearn.metrics import roc_auc_score, recall_score, precision_score,fbeta_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from IPython.display import Image 
from collections import Counter,defaultdict
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mail_attributes import *
from  aa_tp_utils import *
from  sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

ham_txt= json.load(open('./jsons/training_ham.json'))
spam_txt= json.load(open('./jsons/training_spam.json'))
train_data_frame = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail'])
train_data_frame.attributes=list();
train_data_frame['class'] = ['spam' for _ in range(len(spam_txt))]+['ham' for _ in range(len(ham_txt))]
train_data_frame.spam_count = len(train_data_frame[train_data_frame['class'] == 'spam' ])
train_data_frame.ham_count = len(train_data_frame[train_data_frame['class'] == 'ham' ])
train_data_frame.set_name = ''

process_attributes(train_data_frame)
train_data_frame.X = train_data_frame[train_data_frame.attributes].values
train_data_frame.y = train_data_frame['class']
train_data_frame.y_binarized = preprocessing.label_binarize(train_data_frame.y,classes=['ham','spam'])
train_data_frame.y_binarized = np.array([number[0] for number in train_data_frame.y_binarized])

clf = MultinomialNB()
param_grid = {'alpha':[0.0,0.5,1.0],
		'fit_prior' : [ True, False],
		}
f05_score = make_scorer(fbeta_score, beta=0.5)
grid_search = GridSearchCV(clf,param_grid=param_grid, scoring=f05_score,n_jobs=4,cv=5, verbose=True)
grid_search.fit(train_data_frame.X, train_data_frame.y_binarized)


csv_file = open('./plots/cv_grid_search_knn_' + 'f05' + '.csv', "w")
csv_file.write('mean_score,std dev score, n_neighbors,weights, leaf_size, algorithm, p\n')
for test in grid_search.grid_scores_:
	csv_file.write(str(np.mean(test[2]))+','+str(np.std(test[2])))
	for value in test[0].itervalues():
		csv_file.write(',' + str(value))
	csv_file.write('\n')

csv_file.close()

print grid_search.best_estimator_
print grid_search.best_score_

#print 'Testing performance in Testing Data'

# test_data_frame = open_set('jsons/mail_testing_set.json','test_set')
# process_attributes(test_data_frame)
# test_data_frame.X = test_data_frame[test_data_frame.attributes].values
# test_data_frame.y = test_data_frame['class']
# test_data_frame.y_binarized = preprocessing.label_binarize(test_data_frame.y,classes=['ham','spam'])
# test_data_frame.y_binarized = np.array([number[0] for number in test_data_frame.y_binarized])


# y_pred = grid_search.predict(test_data_frame.X)
# print 'fbeta 0.5 on testing = ',fbeta_score(test_data_frame.y_binarized,y_pred,0.5)