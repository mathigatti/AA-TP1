
import re

# Extraigo dos atributos simples: 
# 1) Longitud del mail.
def ma_len(txt):
    return len(txt)

# 2) Cantidad de espacios en el mail.
def ma_count_spaces(txt): 
    return txt.count(" ")

# 3) Simbolo de dolar en txt
def ma_has_dollar(txt): 
    if ('$' in txt):
        return 1
    else:
        return 0

# 4) has link http*://*
def ma_has_link(txt): 
    found = re.match(r'.*https?://.*', txt.replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0

# 5) has HTML
def ma_has_html(txt): 
    found = re.match(r'.*<html>.*</html>.*', txt.lower().replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0

# 6) has CC
def mha_has_cc(txt): 
    found = re.match(r'.*cc:.*', txt.lower().replace('\n','').replace(' ','').replace('bcc:',''))
    if found:
        return 1
    else:
        return 0

# 7) has BCC
def mha_has_bcc(txt): 
    found = re.match(r'.*bcc:.*', txt.lower().replace('\n','').replace(' ',''))
    if found:
        return 1
    else:
        return 0
