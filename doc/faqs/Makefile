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

STYLESHEET=faq.xsl
FAQBOOK=faqbook.xml

all: faqbook.html faq1.html faq2.html printable.html

faqbook.html: ${FAQBOOK} userfaq.xml techfaq.xml ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faqbook.html -PARAM dir 2 -PARAM cont 0 -HTML

faq1.html: userfaq.xml ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faq1.html -PARAM dir 2 -PARAM cont 1 -HTML

faq2.html: techfaq.xml ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT faq2.html -PARAM dir 2 -PARAM cont 2 -HTML

printable.html: ${FAQBOOK} ${STYLESHEET}
	java org.apache.xalan.xslt.Process -IN ${FAQBOOK} -XSL ${STYLESHEET} \
	-OUT printable.html -PARAM dir 2 -PARAM cont print -HTML

clean:
	-rm *.html

veryclean: clean
	-rm *~