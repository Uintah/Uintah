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
#   export CLASSPATH=xt.jar:xp.jar:sax.jar:$CLASSPATH
#

# a gzip'd tarball of the TechnicalGuide in the form of HTML files.
FINALTARGET = TechnicalGuideHtml.tar.gz

# list all of the TechnicalGuide's chapters here, including TOC and print.
# the chapters are named by part and chapter: part 1 chapter 3 = p1c3
# TOC is the Table of Contents and print is the entire book in html, suitable
# for printing from a web browser (Netscape 6 or Explorer 6) to a pdf file.

BASENAMES   = TOC \
              p1c1 p1c2 p1c3 \
              p2c1 p2c2 p2c3 p2c4 p2c5 p2c6 p2c7 p2c8 p2c9 \
              p3c1 \
              p4c1 \
              print

HTMLNAMES   = $(patsubst %,%.html,$(BASENAMES))

default: $(FINALTARGET)

TechnicalGuideHtml.tar.gz: TechnicalGuideHtml.tar
	gzip TechnicalGuideHtml.tar

TechnicalGuideHtml.tar: $(HTMLNAMES)
	cd ..; \
        tar cvf TechnicalGuide/TechnicalGuideHtml.tar TechnicalGuide/*.html

$(HTMLNAMES): TechnicalGuide.xml
	@for i in $(BASENAMES); do \
          echo \
          java com.jclark.xsl.sax.Driver TechnicalGuide.xml \
               ../docbook.xsl $$i.html dir=2 cont=$$i; \
          java com.jclark.xsl.sax.Driver TechnicalGuide.xml \
               ../docbook.xsl $$i.html dir=2 cont=$$i; \
        done

clean: 
	rm -f TechnicalGuideHtml.tar.gz
	rm -f TechnicalGuideHtml.tar
	rm -f $(HTMLNAMES)






