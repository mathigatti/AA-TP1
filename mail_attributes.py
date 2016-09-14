
import re
import pandas as pd
from collections import Counter,defaultdict
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter

global mail_count

mail_count = 0

class MailCorruptedException(Exception):
        pass

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

def word_counter(text,word_count): 
        for word in mail.split():
            word_count[word] += 1

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
def ma_has_cc(mail): 
    found = re.match(r'.*cc:.*', get_mail_headers(mail).lower().replace('\n','').replace(' ','').replace('bcc:',''))
    if found:
        return 1
    else:
        return 0

# 7) has BCC
def ma_has_bcc(mail): 
    found = re.match(r'.*bcc:.*', get_mail_headers(mail).lower().replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0

# 8) has Body
def ma_has_body(mail):
    if get_mail_body(mail) == '':
        return 0
    else:
        return 1
# 9) headers count
def ma_headers_count(headers):
    return len(headers)

# 10) content-type
def ma_categorical_content_type(headers):
    return headers.get('content-type','').split(';')[0].lower().strip()

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

# 13) boundary count

# 14) attachemnet

# 15) attachement type

# 16) Grammar 
    
# 17) Non English characters

# 18) emmbed image ? 

# 19) Mail client x-mailer

# 20)

#Extraccion de palabras mas usadas
def cantApariciones(textos):
    dictTextos = pd.Series({})
    textoLong = 0
    for x in textos:
        palabras = x.split()
        textoLong += len(palabras)
        for y in palabras:
            if y in dictTextos.keys():
                dictTextos[y] += 1.0
            else:
                dictTextos[y] = 1.0
    dictTextos = pesoRelativo(dictTextos,textoLong)

    return dictTextos

def unirApariciones(dict1,dict2):
    for x in dict1.keys():
        if x in dict2.keys():
            dict1[x] = dict1[x] - dict2[x]
        else:
            dict1[x] = dict1[x]
    for x in dict2.keys():
        if not x in dict1.keys():
            dict1[x] = -dict2[x]
    return dict1


def pesoRelativo(textoDicc,textoLong):
    for x in textoDicc.keys():
        textoDicc[x] = textoDicc[x]/textoLong
    return textoDicc

def vector_palabras(texts1,texts2,n):
    dictPalabras1 = cantApariciones(texts1)
    dictPalabras2 = cantApariciones(texts2)

    a = unirApariciones(dictPalabras1,dictPalabras2).sort_values()
    return a[:n] + a[-n:]

def contador(text,nSpam,nHam, bloques):
    longitud = len(text)
    cuantosMailsPorIteracion = int((nSpam+nHam)/bloques)
    for i in range(bloques):
        desdeSpam = int(nHam + nSpam*i/bloques)
        hastaSpam = int(nHam + nSpam*(i+1)/bloques)
        desdeHam = int(nHam*i/bloques)
        hastaHam = int(nHam*(i+1)/bloques)
        print(vector_palabras(text[desdeHam:hastaHam],text[desdeSpam:hastaSpam],10))
