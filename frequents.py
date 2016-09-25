

global palabrasFrecuentes

def esta(word,mail):
    return word in mail

#Lista de las 1000 palabras mas frecuentes 
palabrasFrecuentes = ['lmtpa;',
 # 'abzt',
 # 'x-sieve:',
 # 'revealedto:',
 # 'ambien',
 # 'be3',
 # 'bordercolor=3d#111111',
 # 'v2.2.10)',
 # '(cyrus',
 # 'sha||',
 # 'ce||',
 # '<v:textpath',
 # 'size=3d"2">',
 # '0100x',
 # 'mso-list:l0',
 # '------------a95370304846963',
 # '@2"',
 # '*****on',
 # '72.0pt;layout-grid-mode:ch=',
 # 'requisite*****',
 # '@1"',
 # 'ar;',
 # "style=3d'margin-left:18.0pt;text-indent:-18.0pt;",
 # '#0',
 # '@5"',
 # 'gucci,',
 # 'charset=3dus-ascii">',
 # "mso-layout-grid-align:none'><![if",
 # '------------a40863880133651',
 # 'align=3dcenter><b><font',
 # 'smallcap',
 # 'extender"',
 # 'omega,',
 # 'erectile',
 # 'germany@enron',
 # 'whitt@enron',
 # 'face="',
 # '{font-size:',
 # '99)',
 # '=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=ac=a6=',
 # '888mentors#####-q.com',
 # '(idt)',
 # 'shape=3d"rect"',
 # 'bold;color:',
 # 'border-color:=',
 # '2.25pt;',
 # 'informationdate:',
 # 'so|utions',
 # 'inset;',
 # 'cialis,',
 # 'complications,',
 # '0400x',
 # 'arora@enron',
 # 'watson@enron',
 # 'none"',
 # '+0500x',
 # 'charset=gb2312',
 # 'alt=3d=',
 # 'reply-type=original',
 # 'ce|l',
 # 'consignment',
 # 'material|y',
 # 'der-right-color:rgb(0,51,255);',
 # '10.0pt;',
 # 'border-left-colo=',
 # 'auto"',
 # 'unsuitab|e',
 # 'color=3d#ffffff',
 # 'inc|udes',
 # 'cdgt',
 # 'align=3d"center"><b><font',
 # "style=3d'font:7.0pt",
 # '18.0pt',
 # 'x,p',
 # 'narrow"',
 # '0700from:',
 # 'prescripiton',
 # 'corsiva";',
 # 'comp|eteness.',
 # 'rolex,',
 # 'se|ected',
 # '<paliourg#####>,',
 # '(68',
 # 'align=3d=',
 # '#d4d0c8;',
 # 'color:windowtext;}',
 # 'wrongful|y',
 # 'comp|iance',
 # 'cel|',
 # '%rec_with;',
 # '(freebsd))',
 # 'x*p',
 # 'style=3d"mso-spacerun:',
 # '11pt;',
 # 'style="text-align:',
 # 'bio-matrix',
 # 'eyewear',
 # '0300x',
 # 'guaranteee',
 # '=to:',
 # 'hot-rate',
 # '0"',
 # '0600x',
 # '243)',
 # 'lokay@enron',
 # 'scorman@enron',
 # 'materia||y',
 # 'pharm',
 # 'bgcolor=3d"navy">',
 # '(219',
 # 'xmlns:w=3d"urn:schemas-microsoft-com:office:word"',
 # 'height=3d19>',
 # 'charset="big5"',
 # '(v1.52f)',
 # 'daysto:',
 # 'nbsp',
 # '"forwardlooking',
 # 'materia|ly',
 # 'wel|s',
 # 'we|ls',
 # 'xmlns:o=3d"urn:schemas-microsoft-com:office:office"',
 # 'se|l',
 # 'php-nuke',
 # '"paliourg"',
 # '1.0.92vs)',
 # 'techno|ogies,',
 # 'dfznet',
 # 'style=3d"border-top-width:1;',
 # '<dd>-<font',
 # 'fertility',
 # '+0200x',
 # '381048041@qq.com',
 # 'email.gif"',
 # 'fami|iar',
 # '<381048041@qq.com>',
 # 'exp|oration',
 # '247)',
 # 'size=3d"4"',
 # '700;',
 # '10px',
 # 'charset="charset="gb2312""',
 # 'webuser',
 # 'the-extender',
 # 'pi-lls)',
 # 'margin-top:',
 # 'faithfully,',
 # 'corman@enron',
 # 'boundary="--"',
 # 'class=3dmsonormal><b',
 # '(wgr)',
 # 'wysk.',
 # '(wysk)',
 # 'mso-bidi-font-size:',
 # 'conc|usion:',
 # 'david"><span',
 # '0700x',
 # 'wysk',
 # '27aof',
 # 'invstment',
 # 'securitiesact',
 # 'height=3d19><font',
 # 'size=3d"2"',
 # 'width=3d"700"',
 # 'we|l',
 # 'jasmineupc',
 # 'size=3d1',
 # 'financia|',
 # '#####-face',
 # 'banners=s_from_domain,-,-',
 # '|ease',
 # 'techno|ogies.',
 # '=b8=a6',
 # 'align="middle"',
 # '<##########>;',
 # 'width=3d505',
 # 'sel|',
 # 'coa|bed',
 # 'trave|',
 # '#000000;}',
 # 'x.p',
 # 'shareware',
 # '0.0.0.0)',
 # 'illustrator',
 # 'link=3dblue',
 # 'barre|s',
 # 'class=3dsection1>',
 # '<center>',
 # 'size=3d3',
 # 'dasovich@enron',
 # 'width=3d"0"',
 # 'frzmail',
 # 'style="margin-top:',
 # 'qq.com',
 # "style=3d'font-family:arial;mso-bidi-=",
 # 'x-env-sender:',
 # 'v:shapes=3d"_x=',
 # 'solid"',
 # 'fi|ings',
 # 'smith@enron',
 # 'width=3d"759"',
 # 'x-starscan-version:',
 # 'gb2312',
 # 'we||',
 # 'level2',
 # 'overwhich',
 # "20'ft",
 # 'border-bottom-width:1;',
 # 'panose-1:2',
 # 'se||',
 # 'align=3dleft',
 # 'ms-4b2081e9212e.net',
 # '"ci-ialis',
 # '1200mime',
 # '8pt"',
 # '4522',
 # 'be|iefs,',
 # 'goa|s,',
 # 'viiagrra',
 # 'ci-ialis',
 # 'softabs"',
 # 'e-rectiions',
 # 'haarder',
 # 'solid"=20',
 # 'mso-margin-top-alt:',
 # 'smtpsvc(5.0.2195.6824);',
 # '<nitaigouranga#####>',
 # 'style="mso-bidi-font-weight:',
 # 'se-xual',
 # 'guaaraantees',
 # 'cllick',
 # 'heree:',
 # '17(b),the',
 # '!mso]>',
 # 'p|ans',
 # 'forward-|ooking',
 # '(222',
 # 'style="background-color:',
 # 'dreamwaver',
 # '{color:purple;',
 # 'width:',
 # 'mso-border-alt:',
 # 'height=3d"1"',
 # '.5pt;',
 # 'exp|ode',
 # 'wou|d',
 # 'target=3d"_parent"><img',
 # "eogi's",
 # 'mid-levels',
 # 'borrett',
 # 'be|ieves',
 # 'face="arial',
 # 'comp|ete',
 # 'di|igence',
 # 'ejaculate',
 # 'wrongfu|ly',
 # 'news!!',
 # '=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=eb=a1=',
 # 'width=3d700',
 # '5px',
 # 'em.ca',
 # '+0000from:',
 # 'historica|',
 # 'pr0',
 # 'amatuer',
 # 'class=3dfnb12bl',
 # 'mso-margin-bottom-alt:',
 # '1.0pt',
 # 'bmxg',
 # '+0000date:',
 # 'statements."forward',
 # '"cid:abrptitu.ujnipldk.lhfrvuwh.bfteegsd_csseditor">',
 # '(mos',
 # 'whi|e',
 # '[47',
 # 'exc|usive',
 # 'wel|',
 # 'qzmail',
 # 'http-equiv="content-type"',
 # 'size=3d2>',
 # '1.5pt;',
 # 'va|ue',
 # 'border-top-color:black;',
 # '"f>ree',
 # 'f>ree',
 # 'p.0',
 # 'margin-top:0in&#125;',
 # '18.0pt">f&gt;ree',
 # 'gadgets"',
 # '&#123;mso-style-parent:"";',
 # 'charset=iso-8859-15',
 # 'zoolant.com',
 # 'waterfall,',
 # 'pre-launch',
 # 'align="justify">become',
 # '16.0pt">no',
 # 'size="2">getresponse',
 # 'camera&#146;s,',
 # 'font-family:verdana;',
 # 'src',
 # '(72.26.223.7)',
 # 'gadgets#####',
 # '<gadgets#####>',
 # '3652<br>this',
 # '(i-pods,',
 # '<p><b><font',
 # 'litt|e',
 # 'cellspacing=3d0>',
 # 'align="justify">this',
 # 'cops!!!',
 # '0pt',
 # 'border-left-width:3;',
 # 'hinet',
 # 'affi|iated',
 # '(produced',
 # 'money!!!',
 # '<br><br><table',
 # '|ast',
 # "align='center'><a",
 # "cellpadding='2'",
 # 'colspan=3d"2"',
 # "cellspacing='3'><tr><td",
 # "width='531'",
 # "align='center'",
 # "height='80'",
 # 'borde=',
 # "face='arial,",
 # "border='0'",
 # "size='1'",
 # "sans-serif'><br><img",
 # 'profi|e',
 # 'xmlns:v=3d"urn:schemas-microsoft-com:vml"',
 # 'wrongfu||y',
 # 'high|y',
 # 'id="table1"',
 # 'p|ans,',
 # 'do|lars',
 # 'color="#999999"',
 # '<![endif]-->',
 # '(auth',
 # 'delivery-notification:',
 # 'men)',
 # 'resu|ting',
 # 'urn:content-classes:notice',
 # "width='500'",
 # 'professiona|',
 # 'higher?',
 # 'mnei',
 # '(095)',
 # 'margin:0cm;',
 # 'pub|ic',
 # "wysak's",
 # "5.4pt'>",
 # 'dol|ars',
 # 'netfone',
 # 'original-recipient:',
 # '5.501)',
 # 'qznet',
 # 'mso-pagination:',
 # '$28.45)',
 # 'p.msonormal',
 # 'auto;',
 # '|ose',
 # 'align="center"><font',
 # 'face="tahoma"',
 # 'class="msonormal"',
 # 'eqn=3d"prod',
 # 'comp|eted',
 # 'st0cks',
 # 'height="1"',
 # 'face=3d"arial,',
 # 'border-style:solid;"',
 # '0000date:',
 # 'line-height:',
 # 'width=3d"600"',
 # 'vnbl',
 # 'do||ars',
 # '10.0.2616',
 # 'height=3d"86"',
 # 'boundary="=_nextpart_2rfkindysadvnqw3nerasdf";',
 # '|ocated',
 # '--=_nextpart_2rfkindysadvnqw3nerasdf--',
 # 'nationa|',
 # 'style=3d"border-collapse:',
 # 'width="400"',
 # 'xanax',
 # '(200',
 # 'mil|ion',
 # 'width=3d=',
 # '5.503',
 # 'name=3dgenerator',
 # 'postermime',
 # 'size="3"',
 # 'orgasms',
 # 'comuser',
 # 'globally!',
 # '3o',
 # 'shareho|der',
 # 'a|ready',
 # 'width=3d5',
 # 'abi|ity',
 # 'p|ay',
 '11.0.5510',
 'medica|',
 '"neateye"',
 'color=3d#ff0000',
 'vicodin',
 'border-bottom-color:black;',
 'v:ext=3d"edit"',
 '2616mime',
 '329to:',
 '+0100mime',
 'noting.',
 'verdana,arial,helvetica,sans-serif;font-size:',
 'charset="windows-1251"',
 '|eases',
 'mark.eting',
 'yes">&nbsp;',
 'normal;',
 'style=3d"color:#ffffff;',
 'margin-right:0in;',
 'target=3d"_blank"><img',
 '|arge',
 '<br><br><br>',
 '1o',
 'border-left-width:1;',
 'c_i_a_l_i_s',
 'target="_blank"',
 '<td><div',
 'unreg',
 '2627mime',
 'businessmime',
 'border=3d"1"',
 'hosyou-r02.mine.nu',
 'bold;',
 '--=_nextpart_2rfkindysadvnqw3nerasdf',
 'cd)',
 '21)mime',
 're|ease',
 'face=verdana',
 '200mime',
 'cwtd',
 'showtimes:',
 "netfone's",
 '2462',
 'o-ut',
 'mai-lling',
 'lisst:',
 'pub|ication',
 '2615',
 '118mime',
 'deve|opment',
 'examp|es',
 "x'p",
 'background-color:',
 'v6.00.2800.1106',
 '6700mime',
 'x-originating-email:',
 '52f)',
 '(v1',
 'b|ank',
 '2919',
 '6o',
 'style=3d"font-weight:',
 'size="2"',
 'style="border-right:',
 'sans-serif',
 'width=3d"505"',
 '501)mime',
 'style=3d"background:',
 'present|y',
 'mso-layout-grid-align:',
 'saved!',
 '+0800mime',
 'http-equiv=3dcontent-type',
 'target:',
 'style=3d"border-top-width:0;',
 '(pink',
 'x-source-dir:',
 '1px;',
 'x-source-args:',
 '0)mime',
 'x-source:',
 'current|y',
 'conf|ict',
 '{font-family:',
 '54.7',
 'bighorn',
 'properties;',
 'actua|',
 'align=center><span',
 'mai|',
 '2416',
 '2910',
 'lang=th',
 '=bc=f6',
 'border-bottom-width:0;',
 'border-style:solid;">',
 'text-decoration:',
 'mi|lion',
 'mi||ion',
 '"move","undervalued"',
 'width="1"',
 '"believe",',
 'dir=rtl',
 'width=3d1',
 'face=3d"times',
 'eqn=3d"mid',
 'none"=20',
 'lang=es-mx',
 '<p><b><a',
 'e-savers',
 'chms',
 'corel',
 'carmax.com',
 'size="1"',
 'text-decoration:underline;}',
 'collapse"',
 'aerofoam',
 '<style',
 '"will",',
 "40'ft",
 '(218',
 'bias.the',
 'a|so',
 '0ffer',
 '2730',
 'paliourg',
 'specu|ative',
 'content=3d"microsoft',
 'eqn=3d"if',
 'verdana"',
 'charset="gb2312"',
 "style=3d'mso-bidi-font-weight:normal'><span",
 'width=3d"100%"',
 'additiona|',
 '(211',
 'tadalafil',
 'esmtpa',
 'style="color:',
 '#ece9d8;',
 '5.4pt;',
 'height=3d86',
 'bgcolor=3d#e6e6e6',
 '6.00.2800.1106',
 'extra-time',
 'target=3d"_blank"',
 'color=3d#0000ff',
 'uauthorize',
 'face=3d"arial"',
 'rlsp',
 '(filtered',
 'medium)">',
 'witer',
 '"expect",',
 '(kst)',
 'align=3d"center"><a',
 'http-equiv=3d"content-type"',
 'dir=ltr',
 'inc|uding',
 'cialis.',
 'mai|ings,',
 'border=3d0>',
 'screens;',
 'tadalafil,',
 'border=3d=',
 '+0400reply',
 'on|y',
 'p|aced',
 'format=flowed;',
 'mso-fareast-font-family:',
 'lotto',
 'foreigner',
 'lang=3den-gb',
 'act...',
 'sidebacks',
 '0pt"><span',
 'awesome,',
 '5.4pt',
 'emai|',
 'style=3d"color:',
 'spur-m',
 '"may",',
 '14pt;',
 'alt=3dremail.gif',
 'dir="ltr">',
 'vk.4.04.00.03',
 'type=3d"#_x0000_t75"',
 'padding-bottom:',
 '1]><v:shape',
 'bgcolor=3d#ffffff',
 'align="center"><b>',
 'dfzmail',
 '10pt">',
 'computron',
 '22pt;',
 'border-right-width:=',
 '"intend"',
 '!vml]><img',
 'align=3dcenter><a',
 '0in;',
 'invo|ve',
 '<td><p',
 "align='center'>",
 'v6.00.2900.2180',
 '<td><img',
 "vocalscape's",
 'eogi',
 'internationa|',
 '&#183;',
 '1pt',
 'vocalscape',
 '<br><br><br><br><br><br>',
 'nomad',
 'ar-eg"><span',
 'natura|',
 'align=3d"center"><font',
 'koi8',
 'charset=windows-1251',
 '"will,"',
 'comreply',
 '=mime',
 'margin-bottom:',
 '0cm;',
 'inc|ude',
 '|imited',
 'face=3d"arial',
 'style=3d"font-family:',
 "style=3d'font-size:10.0pt;",
 'microcap',
 '[cn]',
 'vinoble',
 'huifeng',
 'impotence',
 'foxmail',
 'potentia|',
 'padding-top:',
 'dir="ltr"',
 'petro|eum',
 'tahoma;',
 "roman';",
 '<td=20',
 'eqn=3d"sum',
 'face=3d"verdana"',
 'lauraan',
 'postcards',
 'vml',
 'windowtext',
 'resu|ts',
 'valign=3dtop',
 'align=3d"center"',
 '0000mime',
 'vcsc',
 '<div>',
 'erections',
 'fee|',
 'align=3dcenter',
 'cou|d',
 'style="color:#ffffff;',
 'x-message-info:',
 'y:arial;',
 'lang=he',
 'techno|ogy',
 'rx',
 'navy;',
 'bat!',
 'class=3dmsonormal><font',
 'padding-right:',
 'a|l',
 '"projects",',
 'p|ease',
 'align=3d"center">',
 '6.00.2800.1158',
 '(apple',
 'face=3dverdana=20',
 'wi|l',
 'mso-ansi-language:',
 'news|etter',
 'wi||',
 'style=3d"margin:',
 'wil|',
 '1px;">',
 'mso-bidi-language:',
 'align="center">',
 'border-top:',
 '"estimates,"',
 'mso-bidi-font-family:',
 'alt=3d""',
 '<p><a',
 'style="font-family:',
 'style="margin:',
 '"expects",',
 'border-bottom:',
 'face=3darial><span',
 '"foresee",',
 'st0ck',
 '|ooking',
 'border-left:',
 '0000message',
 'face=3dverdana',
 'mso-fareast-language:',
 'text-align:',
 'utf',
 '<td><a',
 '21b',
 'align=3dcenter><font',
 '1px',
 'cellspacing=3d0',
 'verdana;',
 'align=3dmiddle',
 'solid;',
 'face=3darial',
 'cellpadding=3d0',
 '<paliourg#####>',
 '2004))',
 'border=3d0',
 '+0600',
 'oi|',
 '+0500',
 'class=3dmsonormal><span',
 '(ist)',
 'x-antiabuse:',
 'class=3dmsonormal',
 '+0000message',
 '+0400',
 '(squirrelmail',
 '10pt;',
 'cialis',
 'wysak',
 '3ax',
 '3amime',
 'charset="iso-8859-7"',
 'v6.00.2900.2527',
 'border=3d"0"',
 '#####mailserver',
 '0cm',
 'style=3d"font-size:',
 'msmbx03p',
 '8859',
 '0000from:',
 'lang=3den-us',
 '=date:',
 'one-way,',
 'mailserver',
 '+0300mime']

