#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

#
# This file uses xalan-java 2.3.1 or higher to process the xml files.
# The environment variable CLASSPATH must point to xalan.jar, xml-api.jar,
#  and xercesImpl.jar.
#

.PHONY: all

STYLESHEET=../Utilities/XML/faq.xsl
FAQBOOK=faqbook.xml
INSTFAQ=../Installation/FAQ/faq.xml
USERFAQ=../User/FAQ/faq.xml
DEVFAQ=../Developer/FAQ/faq.xml
FAQS=${INSTFAQ} ${USERFAQ} ${DEVFAQ}
DIRLEVEL=1

all: index.html faq1.html faq2.html faq3.html printable.html

index.html: ${FAQBOOK} ${FAQS} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT index.html -PARAM dir ${DIRLEVEL} -PARAM cont 0 -HTML

faq1.html: ${INSTFAQ} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faq1.html -PARAM dir ${DIRLEVEL} -PARAM cont 1 -HTML

faq2.html: ${USERFAQ} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faq2.html -PARAM dir ${DIRLEVEL} -PARAM cont 2 -HTML

faq3.html: ${DEVFAQ} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faq3.html -PARAM dir ${DIRLEVEL} -PARAM cont 3 -HTML

printable.html: ${FAQBOOK} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT printable.html -PARAM dir ${DIRLEVEL} -PARAM cont print -HTML

clean:
	-rm *~

veryclean: clean
	-rm *.html
