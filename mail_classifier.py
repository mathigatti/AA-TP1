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



def split_mail(txt):
    t = tuple(txt.split('\r\n\r\n',1))                          
    if len(t) == 1:                                             
        t = tuple(txt.split('\n\n',1))                          
        if len(t) == 1:                                         
            t = tuple(txt.split('\r\r',1))
            if len(t) == 1:                                     
                t = (txt,'<empty>')                             
    return t[0]                                                 
                                                                
def attribute_ratio(df,attribute):                              
    print attribute                                             
    print sum(df[attribute][:len(ham_txt)])/float(len(ham_txt)) # Armo un dataset de Pandas 
    print sum(df[attribute][len(ham_txt):])/float(len(spam_txt))# http://pandas.pydata.org/

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


    df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
    #df['headers'], df['body']  = zip(*df['text'].map(split_mail))
    df['headers'] = df['text'].map(split_mail)
    df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
    df['len'] = map(len, df.text)
    df['count_spaces'] = map(ma_count_spaces, df.text)

    df['has_dollar'] = map(ma_has_dollar, df.text)
    attribute_ratio(df,'has_dollar')
    df['has_link'] = map(ma_has_link, df.text)
    attribute_ratio(df,'has_link')
    df['has_html'] = map(ma_has_html, df.text)
    attribute_ratio(df,'has_html')
    df['has_cc'] = map(mha_has_cc, df.headers)
    attribute_ratio(df,'has_cc')
    df['has_bcc'] = map(mha_has_bcc, df.headers)
    attribute_ratio(df,'has_bcc')
    

    # Preparo data para clasificar
    X = df[['len', 'count_spaces','has_link','has_dollar','has_html','has_cc','has_bcc']].values
    y = df['class']

    # Elijo mi clasificador.
    clf = DecisionTreeClassifier()

    # Ejecuto el clasificador entrenando con un esquema de cross validation
    # de 10 folds.
    print 'Accuracy: Mean and std dev'
    res = cross_val_score(clf, X, y, cv=10)
    print np.mean(res), np.std(res)
    # salida: 0.783040309346 0.0068052434174  (o similar)

