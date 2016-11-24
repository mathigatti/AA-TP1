import json
import pandas as pd
from mail_attributes import *
from os.path import exists,isfile


def word_counter(text,word_count): 
        for word in text.split():
            word_count[word] += 1

def toString(dictJSON):
	res = []
	for tupla in dictJSON:
		res.append((tupla[0],format(tupla[1], '.2f')))
	return res

#Cuento palabras calcular frecuencia de palabras por clase

if __name__ == '__main__':

	if exists('./jsons/mail_training_set.json') and isfile('./jsons/mail_training_set.json'):
	    df = pd.read_json('./jsons/mail_training_set.json')
	    df.spam_count = len(df[df['class'] == 'spam' ])
	    df.ham_count = len(df[df['class'] == 'ham' ])
	else:
	    ham_txt= json.load(open('./jsons/training_ham.json'))
	    spam_txt= json.load(open('./jsons/training_spam.json'))
	    df = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail'])
	    df['class'] = ['spam' for _ in range(len(spam_txt))]+['ham' for _ in range(len(ham_txt))]
	    df.spam_count = len(df[df['class'] == 'spam' ])
	    df.ham_count = len(df[df['class'] == 'ham' ])

	word_count_ham=defaultdict(int)
	word_count_spam=defaultdict(int)

	map(lambda txt: word_counter(get_mail_body(txt),word_count_ham),df[-df.ham_count:]['raw_mail'])
	word_freq_ham = {k: v / float(df.ham_count) for k, v in word_count_ham.iteritems()}
	map(lambda txt: word_counter(get_mail_body(txt),word_count_spam),df[:df.spam_count]['raw_mail'])
	word_freq_spam = {k: v / float(df.spam_count) for k, v in word_count_spam.iteritems()}


	#HAM Words - dict con las diferencias en cantidad de apariciones en ham y spam. Al final me quedare con las primeras y ultimas mil
	for k, v in word_freq_ham.iteritems():
	    if word_freq_spam.get(k,None) is not None:
	        word_freq_ham[k] = v -  word_freq_spam[k]
	        del word_freq_spam[k]
	    else:
	        word_freq_ham[k] = v 
	for k, v in word_freq_spam.iteritems():
	    word_freq_ham[k] = -v

	word_freq_ham = sorted(word_freq_ham.items(), key=lambda x: x[1])

	text_spam = open("./doc/frequentSpamWords.txt", "w")
	text_ham = open("./doc/frequentHamWords.txt", "w")

	#SPAM Words
	#spam_word_attributes = {k: v for k, v in word_freq_spam.iteritems() if ( (v> 0.5 and  word_freq_ham.get(k,None) is None ) or ( word_freq_ham.get(k,None) is not None and (v - word_freq_ham[k])  > 0.5 ))  }
	text_spam.write("Spam Words: %s" % str(toString(word_freq_ham[0:1000])))
	text_ham.write("ham Words: %s" % str(toString(word_freq_ham[-1000:])))
	text_spam.close()
	text_ham.close()

