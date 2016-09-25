# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score, make_scorer
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus as pydot
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


if __name__ == '__main__':
    if exists('jsons/mail_training_set.json') and isfile('jsons/mail_training_set.json'):
        df = pd.read_json('jsons/mail_training_set.json')
        df.attributes=list();
        df.spam_count = len(df[df['class'] == 'spam' ])
        df.ham_count = len(df[df['class'] == 'ham' ])
        df.set_name = ''
        save_training_test = False
    else:
        ham_txt= json.load(open('./jsons/training_ham.json'))
        spam_txt= json.load(open('./jsons/training_spam.json'))
        df = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail'])
        df.attributes=list();
        df['class'] = ['spam' for _ in range(len(spam_txt))]+['ham' for _ in range(len(ham_txt))]
        df.spam_count = len(df[df['class'] == 'spam' ])
        df.ham_count = len(df[df['class'] == 'ham' ])
        df['mail_headers_dict'] = map(lambda mail: mail_headers_to_dict(get_mail_headers(mail)),df['raw_mail'])
        df['raw_mail_body'] = map(get_mail_body,df['raw_mail'])
        df.set_name = ''
        save_training_test = True
    
    process_attributes(df)

    # Preparo data para clasificar
    X = df[df.attributes].values
    y = df['class']

    # True = Spam, False = Ham
    yBool = booleanizar(y)

    # Elijo mi clasificador.
    dtc0 = DecisionTreeClassifier(class_weight='balanced')
    dtc1 = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=2)
    dtc2 = DecisionTreeClassifier(class_weight='balanced', max_depth=2)
    dtc3 = DecisionTreeClassifier(criterion='entropy', max_depth=2)
    dtc4 = DecisionTreeClassifier(max_depth=2)

    gnb = GaussianNB()
    bnb = BernoulliNB()
    mnb = MultinomialNB()
    svm = svm.SVC()
    rfc = RandomForestClassifier(max_depth=3)

    # Ejecuto el clasificador entrenando con un esquema de cross validation
    # de 10 folds.

    cross_validation_f05('Decision Tree 0', dtc0, X, yBool)

    cross_validation_f05('Decision Tree 1', dtc1, X, yBool)

    cross_validation_f05('Decision Tree 2', dtc2, X, yBool)

    cross_validation_f05('Decision Tree 3', dtc3, X, yBool)

    cross_validation_f05('Decision Tree 4', dtc4, X, yBool)

    cross_validation_f05('Gaussian Naive Bayes', gnb, X, yBool)

    cross_validation_f05('Bernoulli Naive Bayes', bnb, X, yBool)

    cross_validation_f05('Multinomial Naive Bayes', mnb,X,yBool)

    cross_validation_f05('Random Forest', rfc,X,yBool)

#   Faltan SVM, KNN y Arbol de Decision con Pruning como minimo para probar

    if save_training_test == True:
        df[['class','mail_headers_dict','raw_mail_body']].to_json('jsons/mail_training_set.json')

    dtc4.fit(X,y) 
    
    with open("mail_classifier.dot", 'w') as f:
        f = tree.export_graphviz(dtc4,feature_names=df.attributes,out_file=f)


    # scores_max_depth  = defaultdict(list)
    # scorers= ['accuracy','precision','recall','f1','roc_auc']

    # y_binarize = preprocessing.label_binarize(y,classes=['ham','spam'])
    # y_binarize = np.array([number[0] for number in y_binarize])


    # for scorer in scorers:
    #     print '-----------------',scorer,'-----------------'
    #     for i in range(1,40):
    #         clf = DecisionTreeClassifier(class_weight='balanced',max_depth=i)
    #         res = cross_val_score(clf, X, y_binarize, cv=10,scoring=scorer)
    #         scores_max_depth[scorer].append(np.mean(res))
    #         print(i," ",np.mean(res), np.std(res))

    # plt.plot(range(1,40),scores_max_depth['accuracy'])
    # plt.savefig('tree' + '_' +'accuracy' + '_en_funcion_de_'+'max_depth' + '.pdf')
    # plt.close()

    # precision, = plt.plot(range(1,40),scores_max_depth['precision'],label='precision')
    # recall, = plt.plot(range(1,40),scores_max_depth['recall'],label='reacll')
    # plt.legend([precision, recall],['precision','recall'],loc='lower right')
    # plt.savefig('tree' + '_' + 'precision_recall' + '_en_funcion_de_'+'max_depth' + '.pdf')
    # plt.close()

    # knn_scores  = defaultdict(list)

    # for scorer in scorers:
    #     print '-----------------',scorer,'-----------------'
    #     for i in range(1,20):
    #         knn = KNeighborsClassifier(n_neighbors=i)
    #         res = cross_val_score(knn, X, y_binarize, cv=10,scoring=scorer)
    #         knn_scores[scorer].append(np.mean(res))
    #         print(i," ",np.mean(res), np.std(res))

    # plt.plot(range(1,20),knn_scores['accuracy'])
    # plt.savefig('knn' + '_' +'accuracy' + '_en_funcion_de_'+'n_vecinos' + '.pdf')
    # plt.close()

    # precision, = plt.plot(range(1,20),knn_scores['precision'],label='precision')
    # recall, = plt.plot(range(1,20),knn_scores['recall'],label='reacll')
    # plt.legend([precision, recall],['precision','recall'],loc='lower right')
    # plt.savefig('knn' + '_' + 'precision_recall' + '_en_funcion_de_'+'n_vecinos' + '.pdf')
    # plt.close()

