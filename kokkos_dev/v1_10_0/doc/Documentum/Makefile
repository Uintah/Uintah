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
# --
#
# This Makefile uses the Saxon XSLT processor.  CLASSPATH must point to
# saxon.jar
#
# STYLESHEET_PATH must point to the location of the srdocbook stylesheets,
#  e.g. /Applications/docbook-xsl-1.51.1/html/.
#

latex:=latex -interaction=nonstopmode

latex2htmloptions:= -info 0 -split 0 -no_white -link 1 -no_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix module-latex -image_type gif \
-style ../../Utilities/HTML/srlatex2html.css

rmfiles:=rm -f
rmdirs:=rm -rf


default: index.html documentum module-xml module-latex

index.html: index.rhtml ../edition.xml
	eruby --delimiters html $< >$@

documentum : documentum.html

.DELETE_ON_ERROR: documentum.html
documentum.html : documentum.xml
	java com.icl.saxon.StyleSheet -w2 $< ${STYLESHEET_PATH}/srdocbook.xsl >$@

module-xml : module-xml.dvi

module-xml.dvi: module-xml.tex
	-${latex} $<
	-${latex} $<
	dvips -o module-xml.ps -Pps $@
	gzip -c <module-xml.ps >module-xml.ps.gz
	latex2html ${latex2htmloptions} $<


module-latex : module-latex.dvi

module-latex.dvi: module-latex.tex
	-${latex} $<
	-${latex} $<
	dvips -o module-latex.ps -Pps $@
	gzip -c <module-latex.ps >module-latex.ps.gz
	latex2html ${latex2htmloptions} $<

clean:
	-${rmfiles} *.aux *.dvi *.lot *.toc *.log

veryclean: clean
	-${rmfiles} documentum.html *.ps *.gz *~
	-${rmdirs} module-xml module-latex