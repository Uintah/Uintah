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

.PHONY: default

STYLESHEET=../Utilities/XML/faq.xsl
FAQBOOK=faqbook.xml
INSTFAQ=../Installation/FAQ/faq.xml
USERFAQ=../User/FAQ/faq.xml
DEVFAQ=../Developer/FAQ/faq.xml
FAQS=${INSTFAQ} ${USERFAQ} ${DEVFAQ}
TREETOP=../../

#
# If using saxon 6.3 or higher to process the xml files then
# the environment variable CLASSPATH must point to saxon.jar.
#
define xmltohtml
-java com.icl.saxon.StyleSheet -w2 ${FAQBOOK} ${STYLESHEET} \
treetop=${TREETOP} cont=${2} >${1}
endef

# If using xalan-java 2.3.1 or higher to process the xml files then
# the environment variable CLASSPATH must point to xalan.jar, xml-api.jar,
#  and xercesImpl.jar.
#
# define xmltohtml
# -java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
# -OUT ${1} -PARAM treetop ${TREETOP} -PARAM cont ${2} -HTML
# endef

default: index.html faq1.html faq2.html faq3.html printable.html

index.html: ${FAQBOOK} ${FAQS} ${STYLESHEET}
	${call xmltohtml,index.html,0}

faq1.html: ${INSTFAQ} ${STYLESHEET}
	${call xmltohtml,faq1.html,1}

faq2.html: ${USERFAQ} ${STYLESHEET}
	${call xmltohtml,faq2.html,2}

faq3.html: ${DEVFAQ} ${STYLESHEET}
	${call xmltohtml,faq3.html,3}

printable.html: ${FAQBOOK} ${FAQS}
	${call xmltohtml,printable.html,print}

clean:
	-rm *~

veryclean: clean
	-rm *.html
