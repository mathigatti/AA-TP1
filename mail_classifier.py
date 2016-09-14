# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus as pydot
from mail_attributes import *
from collections import Counter,defaultdict
from sklearn import preprocessing
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from os.path import exists,isfile

def add_attribute(data_frame,attribure_name,function,input_attribute,encode=False):
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
        print 'saving json: ' + json_file_path
        data_frame[attribure_name].to_json(json_file_path)

def attribute_ratio(df,attribute):                              
    print(attribute)                                             
    print(sum(df[attribute][:len(ham_txt)])/float(len(ham_txt))) # Armo un dataset de Pandas 
    print(sum(df[attribute][len(ham_txt):])/float(len(spam_txt)))# http://pandas.pydata.org/


if __name__ == '__main__':
    # Leo los mails (poner los paths correctos).
    ham_txt= json.load(open('./jsons/ham_dev.json'))
    spam_txt= json.load(open('./jsons/spam_dev.json'))
    # Imprimo un mail de ham y spam como muestra.
    #print ham_txt[0]
    #print "--------------headers---------------------------------"
    #print ham_txt[0].split('\r\n\r\n',1)[0]
    #print "--------------body----------------------------------------"
    #print ham_txt[0].split('\r\n\r\n',1)[1]
    #print "------------------------------------------------------"
    #print spam_txt[0]
    #print "------------------------------------------------------"


    df = pd.DataFrame(ham_txt+spam_txt, columns=['raw_mail'])
    df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
    
    add_attribute(df,'mail_headers_dict',lambda mail: mail_headers_to_dict(get_mail_headers(mail)),'raw_mail')
    add_attribute(df,'raw_mail_body',get_mail_body,'raw_mail')

    #contador(df.raw_mail, len(spam_txt), len(ham_txt),1000)
    #Cuento palabras calcular frecuencia de palabras por clase
    word_count_ham=defaultdict(int)
    word_count_spam=defaultdict(int)

    #map(lambda txt: mail_word_counter(mail_body(txt),word_count_ham),df.raw_mail[:len(ham_txt)])
    #word_freq_ham = {k: v / float(len(ham_txt)) for k, v in word_count_ham.iteritems()}
    #map(lambda txt: mail_word_counter(mail_body(txt),word_count_spam),df.raw_mail[len(spam_txt)+1:])
    #word_freq_spam = {k: v / float(len(spam_txt)) for k, v in word_count_spam.iteritems()}
    
    
    #HAM Words - dict con palabras que parecen media vez por mail de ham y la palabra no aparece en spam o la diferencia de frecuencia 
    #es mayor a 0.5
    #ham_word_attributes = {k: v for k, v in word_freq_ham.iteritems() if (( v> 0.5 and  word_freq_spam.get(k,None) is None)  or ( word_freq_spam.get(k,None) is not None and (v -  word_freq_spam[k]) > 0.5 ))    }
    #print 'Ham words'
    #print ham_word_attributes
    #SPAM Words - analogo
    #spam_word_attributes = {k: v for k, v in word_freq_spam.iteritems() if ( (v> 0.5 and  word_freq_ham.get(k,None) is None ) or ( word_freq_ham.get(k,None) is not None and (v - word_freq_ham[k])  > 0.5 ))  }
    #print 'Spam words'
    #print spam_word_attributes 

    add_attribute(df,'spell_error_count',lambda mail: ma_spell_error_count(get_mail_body(mail)),'raw_mail')
    add_attribute(df,'raw_mail_len',len,'raw_mail')
    add_attribute(df,'raw_body_count_spaces',ma_count_spaces,'raw_mail_body')
    add_attribute(df,'has_dollar',ma_has_dollar,'raw_mail_body')
    add_attribute(df,'has_link',ma_has_link,'raw_mail_body')
    add_attribute(df,'has_html',ma_has_html,'raw_mail_body')
    add_attribute(df,'has_cc',ma_has_cc,'raw_mail')
    add_attribute(df,'has_bcc',ma_has_bcc,'raw_mail')
    add_attribute(df,'has_body',ma_has_body,'raw_mail')
    add_attribute(df,'headers_count',ma_headers_count,'mail_headers_dict')
    add_attribute(df,'content_type',ma_categorical_content_type,'mail_headers_dict',encode=True)
    add_attribute(df,'recipient_count',ma_recipient_count,'mail_headers_dict')
    attribute_ratio(df,'has_dollar')
    attribute_ratio(df,'has_link')
    attribute_ratio(df,'has_html')
    attribute_ratio(df,'has_cc')
    attribute_ratio(df,'has_bcc')
    attribute_ratio(df,'has_body')
    # Preparo data para clasificar
    X = df[['raw_mail_len', 'raw_body_count_spaces','has_link','has_dollar','has_html','has_cc','has_bcc','has_body','headers_count','content_type','recipient_count','spell_error_count']].values
    y = df['class']

    # Elijo mi clasificador.
    clf = DecisionTreeClassifier()

    # Ejecuto el clasificador entrenando con un esquema de cross validation
    # de 10 folds.
    print('Accuracy: Mean and std dev')
    res = cross_val_score(clf, X, y, cv=10)
    print(np.mean(res), np.std(res))
    # salida: 0.783040309346 0.0068052434174  (o similar)

