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

# this makefile uses the java classes XT and XP by James Clark to convert 
# XML to HTML.  therefore, to use this makefile you must have xt.jar, xp.jar 
# and sax.jar listed in your CLASSPATH, i.e.: 
#
#   export CLASSPATH=./xt.jar:./xp.jar:./sax.jar:$CLASSPATH
#

# a gzip'd tarball of the TechnicalGuide in the form of HTML files.
FINALTARGET = TechnicalGuideHtml.tar.gz

# each chapter (a single .xml file) is converted to a single .html file.
# get list of all chapters (part 1 chapter 3 = p1c3), TOC and print
BASENAMES   = $(shell ls -1 p?c?.xml | sed 's,.xml,,g') TOC printable

#############################################################################

.SUFFIXES:
.SUFFIXES: .xml .html 

HTMLNAMES   = $(patsubst %,%.html,$(BASENAMES) $(OTHERNAMES))

JFLAGS      = com.jclark.xsl.sax.Driver TechnicalGuide.xml

default: $(FINALTARGET)

TechnicalGuideHtml.tar.gz: TechnicalGuideHtml.tar
	rm -f TechnicalGuideHtml.tar.gz
	gzip TechnicalGuideHtml.tar

TechnicalGuideHtml.tar: $(HTMLNAMES)
	rm -f TechnicalGuideHtml.tar
	cd ..; \
        tar cvf TechnicalGuide/TechnicalGuideHtml.tar TechnicalGuide/*.html

.xml.html TOC.html printable.html: ../docbook.xsl ../doc_styles.css
	java $(JFLAGS) ../docbook.xsl $*.html dir=2 cont=$* 

clean: 
	rm -f TechnicalGuideHtml.tar.gz
	rm -f TechnicalGuideHtml.tar
	rm -f $(HTMLNAMES)






