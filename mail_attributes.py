
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
    if get_mail_body(mail) == '':
        return 0
    else:
        return 1
# 9) headers count
def ma_headers_count(headers):
    return len(headers)

# 10) content-type
def ma_categorical_content_type(headers):
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
    content_type = ma_categorical_content_type(headers) 
    found = re.match(r'multipart.*',content_type)
    if found:
        return 1
    else:
        return 0

# 14) parts count
def ma_parts_count(headers,raw_mail_body):
    count = 1
    if ma_is_mulipart(headers) == 1:
        content_type = headers.get('content-type','')
        from_pos = content_type.find('boundary=') + len('boundary=')
        to_pos = from_pos + content_type[from_pos+1:].find('"')
        boundary=content_type[from_pos+1:].strip('"')
        count = len(raw_mail_body.replace('--' + boundary + '--','').split('--'+boundary)) 
    return count
    
    

# 15 ) attachemnet


# ) attachement type

# ) Grammar 
    
# ) Non English characters

# ) emmbed image ? 

# ) Mail client x-mailer

# 20)