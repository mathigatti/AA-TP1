from collections import Counter,defaultdict

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