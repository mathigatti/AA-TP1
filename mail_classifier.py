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
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus as pydot
from mail_attributes import *
from collections import Counter,defaultdict
from sklearn import preprocessing
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from os.path import exists,isfile

def booleanizar(vector):
    yBool = []

    for i in y: 
        if i == 'spam': 
            booleano=True 
        else: 
            booleano=False
        yBool.append(booleano)
    return yBool

def add_attribute_from_series(data_frame,attribure_name,function,input_attribute,encode=False,save=True):
    json_file_path = 'jsons/' + attribure_name + '.json'
    print json_file_path
    if exists(json_file_path) and isfile(json_file_path):
        print 'file exists will load data'
        data_frame[attribure_name] = pd.read_json(json_file_path,typ='series')
    else:
        print 'file does not exists will process data'
        if encode:
            data_frame[attribure_name] = preprocessing.LabelEncoder().fit_transform(data_frame[input_attribute].map(function))
        else:
            data_frame[attribure_name] = data_frame[input_attribute].map(function)
        if save:
            print 'saving json: ' + json_file_path
            data_frame[attribure_name].to_json(json_file_path)

#Allows to work with multiple input attributes, function is df row
def add_attribute_from_df(data_frame,attribure_name,function,encode=False,save=True):
    json_file_path = 'jsons/' + attribure_name + '.json'
    print json_file_path
    if exists(json_file_path) and isfile(json_file_path):
        print 'file exists will load data'
        data_frame[attribure_name] = pd.read_json(json_file_path,typ='series')
    else:
        print 'file does not exists will process data'
        if encode:
            data_frame[attribure_name] = preprocessing.LabelEncoder().fit_transform(data_frame.apply(function,axis=1))
        else:
            data_frame[attribure_name] = data_frame.apply(function,axis=1)
        if save:
            print 'saving json: ' + json_file_path
            data_frame[attribure_name].to_json(json_file_path)

def attribute_ratio(df,attribute):                            
    print attribute
    print '% True for ham: ' + str(sum(df[attribute][:df.ham_count])/float(df.ham_count)*100)
    print '% True for spam: ' + str(sum(df[attribute][df.ham_count+1:])/float(df.spam_count)*100)


if __name__ == '__main__':
    if exists('jsons/mail_training_set.json') and isfile('jsons/mail_training_set.json'):
        df = pd.read_json('jsons/mail_training_set.json')
        df.spam_count = len(df[df['class'] == 'spam' ])
        df.ham_count = len(df[df['class'] == 'ham' ])
        save_training_test = False
    else:
        ham_txt= json.load(open('./jsons/training_ham.json'))
        spam_txt= json.load(open('./jsons/training_spam.json'))
        df = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail'])
        df['class'] = ['spam' for _ in range(len(spam_txt))]+['ham' for _ in range(len(ham_txt))]
        df.spam_count = len(df[df['class'] == 'spam' ])
        df.ham_count = len(df[df['class'] == 'ham' ])
        add_attribute_from_series(df,'mail_headers_dict',lambda mail: mail_headers_to_dict(get_mail_headers(mail)),'raw_mail',save=False)
        add_attribute_from_series(df,'raw_mail_body',get_mail_body,'raw_mail',save=False)
        save_training_test = True

    add_attribute_from_series(df,'spell_error_count',lambda mail: ma_spell_error_count(mail),'raw_mail_body')
    add_attribute_from_series(df,'raw_mail_len',len,'raw_mail')
    add_attribute_from_series(df,'raw_body_count_spaces',ma_count_spaces,'raw_mail_body')
    add_attribute_from_series(df,'has_dollar',ma_has_dollar,'raw_mail_body')
    add_attribute_from_series(df,'has_link',ma_has_link,'raw_mail_body')
    add_attribute_from_series(df,'has_html',ma_has_html,'raw_mail_body')
    add_attribute_from_series(df,'has_cc',ma_has_cc,'mail_headers_dict')
    add_attribute_from_series(df,'has_bcc',ma_has_bcc,'mail_headers_dict')
    add_attribute_from_series(df,'has_body',ma_has_body,'raw_mail_body')
    add_attribute_from_series(df,'headers_count',ma_headers_count,'mail_headers_dict')
    add_attribute_from_series(df,'content_type',ma_categorical_content_type,'mail_headers_dict',encode=True)
    add_attribute_from_series(df,'recipient_count',ma_recipient_count,'mail_headers_dict')
    add_attribute_from_series(df,'is_mulipart',ma_is_mulipart,'mail_headers_dict')
    #df['parts_count'] = df.apply(lambda row:ma_parts_count(row['mail_headers_dict'],row['raw_mail_body']),axis=1)
    add_attribute_from_df(df,'parts_count',lambda row:ma_parts_count(row['mail_headers_dict'],row['raw_mail_body']))
    attribute_ratio(df,'has_dollar')
    attribute_ratio(df,'has_link')
    attribute_ratio(df,'has_html')
    attribute_ratio(df,'has_cc')
    attribute_ratio(df,'has_bcc')
    attribute_ratio(df,'has_body')
    # Preparo data para clasificar
    X = df[['raw_mail_len', 'raw_body_count_spaces','has_link','has_dollar','has_html','has_cc','has_bcc','has_body','headers_count','content_type','recipient_count','spell_error_count','parts_count','is_mulipart','parts_count']].values
    y = df['class']
    # True = Spam, False = Ham
    yBool = booleanizar(y)

    # Elijo mi clasificador.
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

    print('Accuracy Decision Tree Classifier 1: Mean and std dev')
    res1 = cross_val_score(dtc1, X, y, cv=10)
    print(np.mean(res1), np.std(res1))

    print('Accuracy Decision Tree Classifier 2: Mean and std dev')
    res1 = cross_val_score(dtc2, X, y, cv=10)
    print(np.mean(res1), np.std(res1))

    print('Accuracy Decision Tree Classifier 3: Mean and std dev')
    res1 = cross_val_score(dtc3, X, y, cv=10)
    print(np.mean(res1), np.std(res1))

    print('Accuracy Decision Tree Classifier 4: Mean and std dev')
    res1 = cross_val_score(dtc4, X, y, cv=10)
    print(np.mean(res1), np.std(res1))

    print('Accuracy Gaussian Naive Bayes: Mean and std dev')
    res2 = cross_val_score(gnb, X, y, cv=10)
    print(np.mean(res2), np.std(res2))

    print('Accuracy Bernoulli Naive Bayes: Mean and std dev')
    res3 = cross_val_score(bnb, X, y, cv=10)
    print(np.mean(res3), np.std(res3))

    print('Accuracy Multinomial Naive Bayes: Mean and std dev')
    res4 = cross_val_score(mnb, X, y, cv=10)
    print(np.mean(res4), np.std(res4))

    print('Accuracy Random Forest Classifier: Mean and std dev')
    res5 = cross_val_score(rfc, X, y, cv=10)
    print(np.mean(res5), np.std(res5))
    # Pruebo calcular el recall, precision y roc auc para random forest
    print('Recall Random Forest Classifier')
    res6 = cross_val_score(rfc, X, yBool, scoring='recall', cv=10)
    print(np.mean(res6), np.std(res6))
    print('Precision Random Forest Classifier')
    res6 = cross_val_score(rfc, X, yBool, scoring='precision', cv=10)
    print(np.mean(res6), np.std(res6))
    print('ROC AUC Random Forest Classifier')
    res6 = cross_val_score(rfc, X, yBool, scoring='roc_auc', cv=10)
    print(np.mean(res6), np.std(res6))

#    print('Accuracy SVM: Mean and std dev')
#    res5 = cross_val_score(svm, X, y, cv=10)
#    print(np.mean(res5), np.std(res5))

    if save_training_test == True:
        df[['class','mail_headers_dict','raw_mail_body']].to_json('jsons/mail_training_set.json')

