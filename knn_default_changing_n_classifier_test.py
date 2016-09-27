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
import matplotlib.pyplot as plt
from mail_attributes import *
from  aa_tp_utils import *
from  sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
	train_data_frame = open_set('jsons/mail_training_set.json','train_set')

	process_attributes(train_data_frame)
	train_data_frame.X = train_data_frame[train_data_frame.attributes].values
	train_data_frame.y = train_data_frame['class']
	train_data_frame.y_binarized = preprocessing.label_binarize(train_data_frame.y,classes=['ham','spam'])
	train_data_frame.y_binarized = np.array([number[0] for number in train_data_frame.y_binarized])


	scorers={'accuracy':accuracy,'precision':precision,'recall':recall}
	scores  = defaultdict(list)
	max_n=15
	for i in range(1,max_n):
	 	clf = KNeighborsClassifier(n_neighbors=i)
	
		for scorer_name,scorer_func in scorers.iteritems():
			print '-----------------',scorer_name,'-----------------'
			cv_score = cross_val_score(clf, train_data_frame.X, train_data_frame.y_binarized, cv=10,scoring=(scorer_name if scorer_name <> 'f05' else scorer_func) , n_jobs=3)
			scores['cv_' + scorer_name].append(np.mean(cv_score))
			clf = clf.fit(train_data_frame.X,train_data_frame.y)
			pred_y = clf.predict(train_data_frame.X)
			train_score = scorer_func(train_data_frame.y,pred_y)
			scores[train_data_frame.set_name + '_' + scorer_name].append(train_score)
			print i,' ',np.mean(cv_score),' ',np.std(cv_score),' ',train_score

	for i in range(1,max_n):
	 	clf = KNeighborsClassifier(n_neighbors=i)
	 	score = cross_validation_f05('Arbol altura ' + str(i) + ' best',clf,train_data_frame.X,train_data_frame.y_binarized)
	 	scores['cv_' + 'f05'].append(score)
	 	clf = clf.fit(train_data_frame.X,train_data_frame.y)
	 	y_pred = clf.predict(train_data_frame.X)
	 	scores[train_data_frame.set_name + '_' + 'f05'].append(fbeta_score(train_data_frame.y,y_pred,0.5,pos_label='ham'))

	for scorer_name in scorers.iterkeys():
		csv_file = open('./plots/knn_n_' + scorer_name + '.csv', "w")
		csv_file.write('n,taining_data_set,cross_validation\n')
		for i in range(0,max_n-1):
			csv_file.write(str(i+1))
			csv_file.write(',')
			csv_file.write(str(scores[train_data_frame.set_name + '_' + scorer_name][i]))
			csv_file.write(',')
			csv_file.write(str(scores['cv_' + scorer_name][i]))
			csv_file.write('\n')
		csv_file.close()


	csv_file = open('./plots/knn_n_' + 'f05' + '.csv', "w")
	csv_file.write('n,taining_data_set,cross_validation\n')
	for i in range(0,max_n-1):
		csv_file.write(str(i+1))
		csv_file.write(',')
		csv_file.write(str(scores[train_data_frame.set_name + '_' + 'f05'][i]))
		csv_file.write(',')
		csv_file.write(str(scores['cv_' + 'f05'][i]))
		csv_file.write('\n')
	csv_file.close()

	for scorer in scorers.iterkeys():
		plt.xlabel("Vecinos considerados")
		plt.ylabel(scorer)
		cv_score_plt, = plt.plot(range(1,max_n),scores['cv_' + scorer])
		train_score_plt, = plt.plot(range(1,max_n),scores[train_data_frame.set_name + '_' + scorer])
		plt.legend([cv_score_plt, train_score_plt],['Mean value from cross validation','On Training set'],loc='lower right')
		plt.savefig('plots/tree' + '_' + scorer + '_en_funcion_de_'+'vecinos' + '.pdf')
		plt.close()
	
	plt.xlabel("Vecinos considerados")
	plt.ylabel('f05')
	cv_score_plt, = plt.plot(range(1,max_n),scores['cv_' + 'f05'])
	train_score_plt, = plt.plot(range(1,max_n),scores[train_data_frame.set_name + '_' + 'f05'])
	plt.legend([cv_score_plt, train_score_plt],['Mean value from cross validation','On Training set'],loc='lower right')
	plt.savefig('plots/tree' + '_' + 'f05' + '_en_funcion_de_'+'vecinos' + '.pdf')
	plt.close()