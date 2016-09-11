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


    df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
    #df['headers'], df['body']  = zip(*df['text'].map(split_mail))

    #contador(df.text, len(spam_txt), len(ham_txt),1000)
    #Cuento palabras calcular frecuencia de palabras por clase
    word_count_ham=defaultdict(int)
    word_count_spam=defaultdict(int)

    map(lambda txt: mail_word_counter(mail_body(txt),word_count_ham),df.text[:len(ham_txt)])
    word_freq_ham = {k: v / float(len(ham_txt)) for k, v in word_count_ham.iteritems()}
    map(lambda txt: mail_word_counter(mail_body(txt),word_count_spam),df.text[len(spam_txt)+1:])
    word_freq_spam = {k: v / float(len(spam_txt)) for k, v in word_count_spam.iteritems()}
    
    
    #HAM Words - dict con palabras que parecen media vez por mail de ham y la palabra no aparece en spam o la diferencia de frecuencia 
    #es mayor a 0.5
    ham_word_attributes = {k: v for k, v in word_freq_ham.iteritems() if (( v> 0.5 and  word_freq_spam[k] is None)  or (v -  word_freq_spam[k] > 0.5 ))    }
    print 'Ham words'
    print ham_word_attributes
    #SPAM Words - analogo
    spam_word_attributes = {k: v for k, v in word_freq_spam.iteritems() if ( (v> 0.5 and  word_freq_ham[k] is None ) or ( v - word_freq_ham[k]  > 0.5 ))  }
    print 'Spam words'
    print spam_word_attributes 

    df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
    df['len'] = map(len, df.text)
    df['count_spaces'] = map(ma_count_spaces, df.text)
    df['has_dollar'] = map(ma_has_dollar, df.text)
    attribute_ratio(df,'has_dollar')
    df['has_link'] = map(ma_has_link, df.text)
    attribute_ratio(df,'has_link')
    df['has_html'] = map(ma_has_html, df.text)
    attribute_ratio(df,'has_html')
    df['has_cc'] = map(ma_has_cc, df.text)
    attribute_ratio(df,'has_cc')
    df['has_bcc'] = map(ma_has_bcc, df.text)
    attribute_ratio(df,'has_bcc')


    # Preparo data para clasificar
    X = df[['len', 'count_spaces','has_link','has_dollar','has_html','has_cc','has_bcc']].values
    y = df['class']

    # Elijo mi clasificador.
    clf = DecisionTreeClassifier()

    # Ejecuto el clasificador entrenando con un esquema de cross validation
    # de 10 folds.
    print('Accuracy: Mean and std dev')
    res = cross_val_score(clf, X, y, cv=10)
    print(np.mean(res), np.std(res))
    # salida: 0.783040309346 0.0068052434174  (o similar)

