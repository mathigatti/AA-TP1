import json
import numpy as np
import pandas as pd
from os.path import exists
from sys import argv,exit
from mail_utils import *

if __name__ == '__main__':
	if len(argv) < 4:
		print 'Usage: '+ argv[0] + ' <ham_json> <spam_json> <outputjson>'
		exit(-2)
	if not exists(argv[3]):
		ham_txt = json.load(open(argv[1]))
		spam_txt = json.load(open(argv[2]))
		df = pd.DataFrame(spam_txt+ham_txt, columns=['raw_mail'])
		df['class'] = ['spam' for _ in range(len(spam_txt))]+['ham' for _ in range(len(ham_txt))]
		df.spam_count = len(df[df['class'] == 'spam' ])
		df.ham_count = len(df[df['class'] == 'ham' ])
		df['mail_headers_dict'] = map(lambda mail: mail_headers_to_dict(get_mail_headers(mail)),df['raw_mail'])
		df['raw_mail_body'] = map(get_mail_body,df['raw_mail'])
		print 'Saving to ' + argv[3]
		df[['class','mail_headers_dict','raw_mail_body']].to_json(argv[3])
	else:
		print'Training set file: ' +  argv[3] + ' already exists'
	print 'Bye!'