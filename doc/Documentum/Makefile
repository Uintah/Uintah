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
#
# The following env variables must be set:
#
# CLASSPATH: Points to saxon.jar.
# STYLESHEET_XSL_HTML: Points at <docbook xsl distro>/html
# STYLESHEET_DSSSL_PRINT: Points at <docbook dsssl distro>/print
# XML_DCL: Points at xml.dcl
# CATALOG: Points at the catalog file that in turn points to
#  .../openjade/dsssl/catalog and .../ISOEnts/isoent.cat (Ugh!).
#

params_common=
chunk_root := documentum.chunk
parms_chunk=${params_common} chunk.section.depth=1 use.id.as.filename=1 \
  root.filename=${chunk_root}
params_nochunk=${params_common} 
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
latex:=latex -interaction=nonstopmode

latex2htmloptions:= -info 0 -split 0 -no_white -link 1 -no_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix module-latex -image_type gif \
-style ../../Utilities/HTML/srlatex2html.css

rmfiles:=rm -f
rmdirs:=rm -rf

edition := ../edition.xml

.PHONY: \
  default \
  documentum-chunk \
  documentum-nochunk \
  documentum-print \
  clean \
  veryclean

.DELETE_ON_ERROR:

default: \
  index.html \
  documentum-chunk \
  documentum-nochunk \
  documentum-print \
  module-xml \
  module-latex

index.html: index.rhtml ${edition}
	eruby --delimiters html $< >$@

documentum : documentum-chunk documentum-nochunk documentum-print

documentum-chunk : ${chunk_root}.html

documentum.html.xml : documentum.rxml ${edition}
	OUTPUT_MODE=html eruby --delimiters xml documentum.rxml > $@

documentum.print.xml : documentum.rxml ${edition}
	OUTPUT_MODE=print eruby --delimiters xml documentum.rxml > $@

${chunk_root}.html : documentum.html.xml
	java com.icl.saxon.StyleSheet -w2 $< \
         ${stylesheet_chunk} ${parms_chunk}

documentum-nochunk : documentum.html

documentum.html : documentum.html.xml
	java com.icl.saxon.StyleSheet -w2 $< \
         ${stylesheet_nochunk} ${parms_nochunk} >$@

documentum-print : documentum.pdf

documentum.pdf : documentum.dvi
	dvipdfm -p letter $<

documentum.dvi : documentum.tex
	for i in 1 2 ; do jadetex documentum.tex ; done

documentum.tex : documentum.print.xml
	openjade -t tex -V tex-backend -c ${CATALOG} -o documentum.tex \
-d ${stylesheet_print} ${XML_DCL} documentum.print.xml 

module-xml : module-xml.dvi

module-xml.dvi: module-xml.tex
	-${latex} $<
	-${latex} $<
	dvipdfm -p letter ${basename $<}
	latex2html ${latex2htmloptions} $<

module-latex : module-latex.dvi

module-latex.dvi: module-latex.tex
	-${latex} $<
	-${latex} $<
	dvipdfm -p letter ${basename $<}
	latex2html ${latex2htmloptions} $<

clean:
	-${rmfiles} *.aux *.dvi *.lot *.toc *.log *~

veryclean: clean
	-${rmfiles} documentum.*.xml *.html *.pdf
	-${rmdirs} module-xml module-latex