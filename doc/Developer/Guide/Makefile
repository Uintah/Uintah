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

# All the html files that make up the book.
BASENAMES   = $(shell ls -1 p?c?.xml | sed 's,.xml,,g') TOC printable
HTMLNAMES  = $(patsubst %,%.html,$(BASENAMES))
BOOK = DeveloperGuide.xml
STYLESHEET =../../Utilities/XML/docbook-xsl/html/docbook.xsl

define xmltohtml
java org.apache.xalan.xslt.Process -IN ${BOOK} -XSL ${STYLESHEET} \
-OUT $@ -PARAM dir 2 -PARAM cont ${patsubst %.html,%,$@} -HTML
endef

default: $(HTMLNAMES)

%.html: %.xml ${STYLESHEET}
	${xmltohtml}

TOC.html printable.html: ${STYLESHEET} ${BOOK}
	${xmltohtml}

clean: 
	rm *~

veryclean: clean
	rm -f $(HTMLNAMES)
