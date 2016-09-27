# -*- coding: utf-8 -*-
import re
import pandas as pd
from collections import Counter,defaultdict
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from mail_utils import *
from nltk.corpus import words
from nltk import word_tokenize,pos_tag

global english_dict

english_dict = set(words.words())

def ma_spell_error_count(raw_mail_body):
    global english_dict
    raw_mail_body = raw_mail_body.lower()
    if ma_has_html(raw_mail_body):
        raw_mail_body = re.sub(r'content-type.*?LINEFEED','',raw_mail_body.replace('\n','LINEFEED').replace('\r',''))
        raw_mail_body = re.sub(r'</?.*?>','',raw_mail_body.replace('LINEFEED',' ').replace('\r',''))
        raw_mail_body = re.sub(r'-|\?|\.|&|#|\$|%|_|!|\"|\'|,|=|\)|\(|}|{|:',' ',raw_mail_body)
    count = 0
    for word in raw_mail_body.split():
        try:
            if word not in english_dict:
                count += 1
        except:
            print 'exception'
    return count


### helper functions
def get_new_line_code(text):
    if text.find('\r\n') >= 0 :
        return '\r\n'
    elif text.find('\r') >= 0 :
        return '\r'
    elif text.find('\n') >= 0 :
        return '\n'
    else:
        raise MailCorruptedException('Not able to identify new line')

def get_mail_headers(mail):
    header_end = mail.find('\r\n\r\n')
    if header_end < 0:
        header_end = mail.find('\n\n')
        if header_end < 0:
             header_end = mail.find('\r\r')
    if header_end < 0:
        headers = mail
    else:
        headers = mail[:header_end]
    return headers

def mail_headers_to_dict(headers):
    headers_dict = defaultdict(str)
    nl = get_new_line_code(headers)
    hdr_list = headers.split(nl)
    for line in hdr_list:
        if line.find(':') >= 0:
            t = line.split(':',1)
            headers_dict[t[0].lower()] = t[1]
            last_header = t[0].lower()
        else:
            headers_dict[last_header] += line;
    return headers_dict

def get_mail_body(mail):
    t = tuple(mail.split('\r\n\r\n',1))
    if len(t) == 1:
        t = tuple(mail.split('\n\n',1))
        if len(t) == 1:
            t = tuple(mail.split('\r\r',1))
            if len(t) == 1:
                t = (mail,'')
    return t[1]


### ATRIBUTOS
# 1) Longitud del mail.
def ma_len(mail):
    return len(mail)

# 2) Cantidad de espacios en el mail.
def ma_count_spaces(mail): 
    return mail.count(" ")

# 3) Simbolo de dolar en mail
def ma_has_dollar(mail): 
    if ('$' in mail):
        return 1
    else:
        return 0

# 4) has link http*://*
def ma_has_link(mail): 
    found = re.match(r'.*https?://.*', mail.replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0

# 5) has HTML
def ma_has_html(mail): 
    found = re.match(r'.*<html>.*</html>.*', mail.lower().replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0

# 6) has CC
def ma_has_cc(headers): 
    if headers.get('cc','') == '':
        return 0
    else:
        return 1

# 7) has BCC
def ma_has_bcc(headers): 
    if headers.get('bcc','') == '':
        return 0
    else:
        return 1

# 8) has Body
def ma_has_body(mail):
    if mail == '':
        return 0
    else:
        return 1

# 9) headers count
def ma_headers_count(headers):
    return len(headers)

# 10) content-type
def ma_content_type(headers):
    return headers.get('content-type','').split(';')[0].lower().strip().split(' ')[0]

# 11) receipients_count
def ma_recipient_count(headers):
    count = 0
    to = headers.get('to','')
    if to !=  '':
       count += len(to.split(',')) 
    cc = headers.get('cc','')
    if cc <> '':
       count += len(cc.split(','))
    bcc = headers.get('bcc','')
    if bcc <> '':
       count += len(bcc.split(',')) 
    return count

# 12) Spell Error count
def ma_spell_error_count(mail):
    spell_checker = SpellChecker("en_US",filters=[EmailFilter,URLFilter])
    count = 0
    spell_checker.set_text(mail)
    for err in spell_checker:
        count += 1
    return count

# 13) is multipart
def ma_is_mulipart(headers):
    content_type = ma_content_type(headers) 
    found = re.match(r'multipart.*',content_type)
    if found:
        return 1
    else:
        return 0

# 14) parts count
def ma_parts_count(headers,raw_mail_body):
    count = 0
    if ma_is_mulipart(headers) == 1:
        content_type = headers.get('content-type','')
        from_pos = content_type.find('boundary=') + len('boundary=')
        to_pos = from_pos + content_type[from_pos+1:].find('"')
        boundary=content_type[from_pos+1:].strip('"')
        count = len(raw_mail_body.replace('--' + boundary + '--','').split('--'+boundary)) 
    return count
    
    

# 15 ) attachemnet
def ma_has_attachment(headers,raw_mail_body):
    if ma_is_mulipart(headers) == 1:
        content_type = headers.get('content-type','')
        from_pos = content_type.find('boundary=') + len('boundary=')
        to_pos = from_pos + content_type[from_pos+1:].find('"')
        boundary=content_type[from_pos+1:].strip('"')
        for part in raw_mail_body.replace('--' + boundary + '--','').split('--'+boundary):
            if part.find('content-type') > 0 and ( part.find('/html') < 0 or part.find('text/') <0 ):
                return 1
    return 0

# 16 ) has_word
def ma_word_count(word,raw_mail_body):
    return raw_mail_body.lower().count(' '+ word + ' '  )


# 17) Uppercase count
def ma_uppercase_count(raw_mail_body):
    return  sum(1 for c in raw_mail_body if c.isupper())

# 18 ) Has Non English characters
def ma_has_non_english_chars(raw_mail_body):
    try:
        raw_mail_body.decode('ascii')
    except:
        return 1
    else:
        return 0

# 19 ) Mail client x-mailer
def ma_mailer(headers): 
    return headers.get('x-mailer','undefined')

# 20 ) subject length
def ma_subject_length(headers): 
    return len(headers.get('subject',''))

# 21) content-transfer-encoding 
def ma_content_transfer_encoding(headers): 
    return headers.get('content-transfer-encoding','undefined')
    
# 22) spaces ratio on body
def ma_spaces_over_len(raw_mail_body):
    if raw_mail_body <> '':
        return (raw_mail_body.count(' ') / float(len(raw_mail_body)))
    else:
        return 0

# 23) lexical diversity
def ma_lexical_diversity(raw_mail_body):
    print raw_mail_body
    if ma_has_html(raw_mail_body):
        raw_mail_body = re.sub(r'content-type.*?LINEFEED','',raw_mail_body.replace('\n','LINEFEED').replace('\r',''))
        raw_mail_body = re.sub(r'</?.*?>','',raw_mail_body.replace('LINEFEED',' ').replace('\r',''))
        raw_mail_body = re.sub(r'-|\?|\.|&|#|\$|%|_|!|\"|\'|,|=|\)|\(|}|{|:',' ',raw_mail_body)
    if  ma_has_body(raw_mail_body) == 0:
        return 0
    return len(set(raw_mail_body).split(' ')) / len(raw_mail_body)

# 24 ) has_word
def ma_has_word(word,raw_mail_body):
    if raw_mail_body.lower().find(word) >= 0:
        return 1
    else:
        return 0