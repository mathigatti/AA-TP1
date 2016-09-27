from mail_attributes import *
from os.path import exists,isfile
from sklearn import preprocessing
from frequents import palabrasFrecuentes
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score, make_scorer, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split
from frequents import palabrasFrecuentes,esta
from frequents2 import palabrasFrecuentes2
from frequents3 import palabrasFrecuentes3
import numpy as np
from frequent_spam_words import *

def open_set(path,set_name):
    if exists(path) and isfile(path):
        df = pd.read_json(path)
        df.attributes=list();
        df.spam_count = len(df[df['class'] == 'spam' ])
        df.ham_count = len(df[df['class'] == 'ham' ])
        df.set_name = set_name
        return df
    else:
        raise Exception('mail_testing_set.json not found\You can use consolidate_input_set.py to create it.\n')

#Procesa tolas las funciones que se llaman ma_*
# El nombre del atributo es lo que sigue a ma_
def process_attributes(df):
    # add_attribute_from_series(df,'spell_error_count',lambda mail: ma_spell_error_count(mail),'raw_mail_body')
    add_attribute_from_series(df,'raw_mail_len',len,'raw_mail_body')
    add_attribute_from_series(df,'raw_body_count_spaces',ma_count_spaces,'raw_mail_body')
    add_attribute_from_series(df,'has_dollar',ma_has_dollar,'raw_mail_body')
    add_attribute_from_series(df,'has_link',ma_has_link,'raw_mail_body')
    add_attribute_from_series(df,'has_html',ma_has_html,'raw_mail_body')
    add_attribute_from_series(df,'has_cc',ma_has_cc,'mail_headers_dict')
    add_attribute_from_series(df,'has_bcc',ma_has_bcc,'mail_headers_dict')
    add_attribute_from_series(df,'has_body',ma_has_body,'raw_mail_body')
    add_attribute_from_series(df,'headers_count',ma_headers_count,'mail_headers_dict')
    add_attribute_from_series(df,'content_type',ma_content_type,'mail_headers_dict',encode=True)
    add_attribute_from_series(df,'recipient_count',ma_recipient_count,'mail_headers_dict')
    add_attribute_from_series(df,'is_mulipart',ma_is_mulipart,'mail_headers_dict')
    add_attribute_from_series(df,'uppercase_count',ma_uppercase_count,'raw_mail_body')
    add_attribute_from_series(df,'has_non_english_chars',ma_has_non_english_chars,'raw_mail_body')
    add_attribute_from_series(df,'mailer',ma_mailer,'mail_headers_dict',encode=True)
    add_attribute_from_series(df,'subject_length',ma_subject_length,'mail_headers_dict')
    add_attribute_from_series(df,'content_transfer_encoding,',ma_content_transfer_encoding,'mail_headers_dict',encode=True)
    add_attribute_from_series(df,'spaces_over_len',ma_spaces_over_len,'raw_mail_body')
    #add_attribute_from_series(df,'word_count',ma_word_count,'raw_mail_body')
    #add_attribute_from_series(df,'avg_word_len',ma_avg_word_len,'raw_mail_body')
    

    #df['parts_count'] = df.apply(lambda row:ma_parts_count(row['mail_headers_dict'],row['raw_mail_body']),axis=1)
    add_attribute_from_df(df,'parts_count',lambda row:ma_parts_count(row['mail_headers_dict'],row['raw_mail_body']))
    add_attribute_from_df(df,'has_attachment',lambda row:ma_has_attachment(row['mail_headers_dict'],row['raw_mail_body']))

    for word in ['a', 'and', 'for', 'of', 'to', 'in', 'the']:
        print word
        add_attribute_from_series(df,word,lambda raw_mail_body: ma_word_count(word,raw_mail_body),'raw_mail_body')
    for word in frequentSpamWords:
        add_attribute_from_series(df,word,lambda raw_mail_body: ma_has_word(word,raw_mail_body),'raw_mail_body')
    #for palabraFrecuente in palabrasFrecuentes:
    #    add_attribute_from_series(df,'esta_' + palabraFrecuente + '_?',lambda mail: esta(palabraFrecuente,mail),'raw_mail_body')    
    # for palabraFrecuente in palabrasFrecuentes2:
    #     add_attribute_from_series(df,'esta_' + palabraFrecuente + '_?',lambda mail: esta(palabraFrecuente,mail),'raw_mail_body')    
    # for palabraFrecuente in palabrasFrecuentes3:
    #     add_attribute_from_series(df,'esta_' + palabraFrecuente + '_?',lambda mail: esta(palabraFrecuente,mail),'raw_mail_body')    

def booleanizar(y):
    yBool = []

    for i in y: 
        if i == 'spam': 
            booleano=True 
        else: 
            booleano=False
        yBool.append(booleano)
    return yBool

def add_attribute_from_series(data_frame,attribure_name,function,input_attribute,set_name='',encode=False,save=True):


    json_file_path = 'jsons/' + ('' if data_frame.set_name=='' else data_frame.set_name + '_') + attribure_name + '.json'
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
    data_frame.attributes.append(attribure_name)

#Allows to work with multiple input attributes, function is df row
def add_attribute_from_df(data_frame,attribure_name,function,set_name='',encode=False,save=True):
    json_file_path = 'jsons/' + ('' if data_frame.set_name=='' else data_frame.set_name + '_') + attribure_name + '.json'   
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
    data_frame.attributes.append(attribure_name)

def attribute_ratio(df,attribute):                            
    print attribute
    print '% True for ham: ' + str(sum(df[attribute][:df.ham_count])/float(df.ham_count)*100)
    print '% True for spam: ' + str(sum(df[attribute][df.ham_count+1:])/float(df.spam_count)*100)


def precision(y_true,y_pred):
    return precision_score(y_true,y_pred,pos_label='spam')

def recall(y_true,y_pred):
    return recall_score(y_true,y_pred,pos_label='spam')

def accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)


def cross_validation_f05(nombre,metodo,x,y):
    f05_score = make_scorer(fbeta_score, beta=0.5)
    res = cross_val_score(metodo, x, y, scoring=f05_score, cv=10, n_jobs=3)
    print nombre + ': Mean and Standard Deviation'
    print(np.mean(res), np.std(res))
    return np.mean(res)