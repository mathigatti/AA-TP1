
import re

### helper functions
def get_headers(mail):
    t = tuple(mail.split('\r\n\r\n',1))
    if len(t) == 1:
        t = tuple(mail.split('\n\n',1))  
        if len(t) == 1:
            t = tuple(mail.split('\r\r',1))  
            if len(t) == 1:
                t = (mail,'')
    return t[0]

def mail_body(mail):
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
def ma_has_cc(mail): 
    found = re.match(r'.*cc:.*', get_headers(mail).lower().replace('\n','').replace(' ','').replace('bcc:',''))
    if found:
        return 1
    else:
        return 0

# 7) has BCC
def ma_has_bcc(mail): 
    found = re.match(r'.*bcc:.*', get_headers(mail).lower().replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0
