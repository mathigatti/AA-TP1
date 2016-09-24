# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus as pydot
from collections import Counter,defaultdict
from sklearn import preprocessing
from os.path import exists,isfile
import matplotlib.pyplot as plt
from mail_attributes import *
from  aa_tp_utils import *


if __name__ == '__main__':
	train_df = open_set('jsons/mail_training_set.json','test_set')
	test_df = open_set('jsons/mail_testing_set.json','train_set')

	process_attributes(train_df)
	train_df.X = train_df[train_df.attributes].values
	train_df.y = train_df['class']
	train_df.y_binarized = preprocessing.label_binarize(train_df.y,classes=['ham','spam'])
	train_df.y_binarized = np.array([number[0] for number in train_df.y_binarized])


	process_attributes(test_df)
	test_df.X = test_df[test_df.attributes].values
	test_df.y = test_df['class']
	test_df.y_binarized = preprocessing.label_binarize(test_df.y,classes=['ham','spam'])
	test_df.y_binarized = np.array([number[0] for number in test_df.y_binarized])

	scorers=['accuracy','precision','recall','f1','roc_auc']
	scores_max_depth  = defaultdict(list)


	for scorer in ['accuracy']:
		print '-----------------',scorer,'-----------------'
		for i in range(1,30):
		    clf = DecisionTreeClassifier(class_weight='balanced',max_depth=i)
		    cv_score = cross_val_score(clf, train_df.X, train_df.y_binarized, cv=10,scoring=scorer)
		    scores_max_depth['cv_' + scorer].append(np.mean(cv_score))
		    clf.fit(train_df.X,train_df.y)
		    train_score = clf.score(train_df.X,train_df.y)
		    scores_max_depth[train_df.set_name + '_' + scorer].append(train_score)
		    test_score = clf.score(test_df.X,test_df.y)
		    scores_max_depth[test_df.set_name + '_' + scorer].append(test_score)
		    print i,' ',np.mean(cv_score),' ',np.std(cv_score),' ',train_score,' ',test_score


	plt.xlabel("Altura maxina del arbol")
	plt.ylabel("Accuracy")
	cv_accuracy, = plt.plot(range(1,30),scores_max_depth['cv_accuracy'])
	train_accuracy, = plt.plot(range(1,30),scores_max_depth[train_df.set_name + '_accuracy'])
	test_accuracy, = plt.plot(range(1,30),scores_max_depth[test_df.set_name + '_accuracy'])
	plt.legend([cv_accuracy, train_accuracy],['Mean value from cross validation','On Training set'],loc='lower right')
	plt.savefig('tree' + '_' + 'accuracy' + '_en_funcion_de_'+'max_depth' + '.pdf')
	plt.close()
