
import re
import pandas as pd

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
