import json
import pandas as pd
from mail_classifier import *
from mail_attributes import *
from os.path import exists,isfile

global spell_checker

def validString(s):
	if isinstance(s, str):
		return True
	elif isinstance(s, unicode):
		return True
	else:
		return False

def word_counter(raw_mail_body,word_count): 
	raw_mail_body = raw_mail_body.lower()
	if ma_has_html(raw_mail_body):
		raw_mail_body = re.sub(r'content-type.*LINEFEED','',raw_mail_body.replace('\n','LINEFEED').replace('\r',''))
		raw_mail_body = re.sub(r'</?.*>','',raw_mail_body.replace('LINEFEED','').replace('\r',''))
		raw_mail_body = re.sub(r'-|\?|\.|&|#|\$|%|_|!|\"|\'|,',' ',raw_mail_body)
		
	for word in raw_mail_body.split():
		try:
			if validString(word) and spell_checker.check(word):
				word_count[word] += 1
		except:
			pass

def toString(dictJSON):
	res = []
	for tupla in dictJSON:
		res.append((tupla[0],format(tupla[1], '.2f')))
	return res

#Cuento palabras calcular frecuencia de palabras por clase
if __name__ == '__main__':
	spell_checker = SpellChecker("en_US",filters=[EmailFilter,URLFilter])
	if exists('./jsons/mail_training_set.json') and isfile('./jsons/mail_training_set.json'):
	    df = pd.read_json('./jsons/mail_training_set.json')
	else:
	    ham_txt= json.load(open('./jsons/training_ham.json'))
	    spam_txt= json.load(open('./jsons/training_spam.json'))
	    df = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail_body'])

	df.spam_count = len(df[df['class'] == 'spam' ])
	df.ham_count = len(df[df['class'] == 'ham' ])

	word_count_ham=defaultdict(int)
	word_count_spam=defaultdict(int)

	map(lambda txt: word_counter(txt,word_count_ham),df[df['class']=='ham']['raw_mail_body'])
	word_freq_ham = {k: v / float(df.ham_count) for k, v in word_count_ham.iteritems()}
	map(lambda txt: word_counter(txt,word_count_spam),df[df['class']=='spam']['raw_mail_body'])
	word_freq_spam = {k: v / float(df.spam_count) for k, v in word_count_spam.iteritems()}


	#HAM Words - dict con las diferencias en cantidad de apariciones en ham y spam. Al final me quedare con las primeras y ultimas mil
	print sorted(word_freq_spam.items(), key=lambda x: x[1])[-50:]
	for k, v in word_freq_ham.iteritems():
	    if word_freq_spam.get(k,None) is not None:
	        word_freq_ham[k] = abs(v -  word_freq_spam[k])
	        del word_freq_spam[k]
	    else:
	        word_freq_ham[k] = v 
	for k, v in word_freq_spam.iteritems():
	    word_freq_ham[k] = v

	frequent_words  =  {k: v for k, v in word_freq_ham.items() if v > 2}.keys()

	with open('jsons/frequent.json', 'w') as fp:
	    json.dump(frequent_words, fp)