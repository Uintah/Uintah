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

# Build print and online versions of the documentum, module-xml, and
# module-latex documents.

chunk_root := documentum.chunk
parms_chunk=chunk.section.depth=1 use.id.as.filename=1 \
  root.filename=${chunk_root}
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
params_print := -V %footnote-ulinks% -V %section-autolabel% \
  -V "(define %default-quadding% 'justify)"
latex:=latex -interaction=nonstopmode
latex2htmloptions:= -info 0 -split 0 -no_white -link 1 -no_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix module-latex -image_type gif \
-style ../../Utilities/HTML/srlatex2html.css

define rxml_to_pxml
OUTPUT_MODE=print eruby --delimiters xml ${1} > ${subst .rxml,.pxml,${1}}
endef

define rxml_to_hxml
OUTPUT_MODE=html eruby --delimiters xml ${1} > ${subst .rxml,.hxml,${1}}
endef

rmfiles:=rm -f
rmdirs:=rm -rf

edition := ../edition.xml

.PHONY: \
  all \
  documentum-html \
  documentum-print \
  module-xml-html \
  module-xml-print \
  module-latex-html \
  module-latex-print \
  clean \
  veryclean
#  component-html \
#  component-print

.DELETE_ON_ERROR:

all: \
  index.html \
  documentum-html \
  documentum-print \
  module-xml-html \
  module-xml-print \
  module-latex-html \
  module-latex-print
#  component-html \
#  component-print

# Rule for building index.html

index.html: index.rhtml ${edition}
	eruby --delimiters html $< >$@

# Rules for building documentum.

documentum : documentum-html documentum-print

documentum-html : documentum.html

documentum.html : documentum.rxml ${edition}
	${call rxml_to_hxml,$<}
	java com.icl.saxon.StyleSheet -w2 documentum.hxml \
         ${stylesheet_chunk} ${parms_chunk}
	java com.icl.saxon.StyleSheet -w2 documentum.hxml \
         ${stylesheet_nochunk} ${parms_nochunk} >$@

documentum-print : documentum.pdf

documentum.pdf : documentum.rxml ${edition}
	${call rxml_to_pxml,$<}
	openjade -t tex -V tex-backend -c ${CATALOG} -o documentum.tex \
-d ${stylesheet_print}  ${params_print} ${XML_DCL} documentum.pxml 
	for i in 1 2 ; do jadetex documentum.tex ; done
	dvipdfm -p letter documentum.dvi

# Rules for building component reference.

component : component-html component-print

component-html : component.html

component.html : component.rxml ${edition}
	${call rxml_to_hxml,$<}
	java com.icl.saxon.StyleSheet -w2 component.hxml \
         ${stylesheet_chunk} ${parms_chunk}
	java com.icl.saxon.StyleSheet -w2 component.hxml \
         ${stylesheet_nochunk} ${parms_nochunk} >$@

component-print : component.pdf

component.pdf : component.rxml ${edition}
	${call rxml_to_pxml,$<}
	openjade -t tex -V tex-backend -c ${CATALOG} -o component.tex \
-d ${stylesheet_print}  ${params_print} ${XML_DCL} component.pxml 
	for i in 1 2 ; do jadetex component.tex ; done
	dvipdfm -p letter component.dvi

# Rules for building module-xml docs

module-xml-html: module-xml/index.html

module-xml-print : module-xml.pdf 

module-xml.dvi: module-xml.tex
	-${latex} $<
	-${latex} $<

module-xml.pdf: module-xml.dvi
	dvipdfm -p letter ${basename $<}

module-xml/index.html: module-xml.dvi
	latex2html ${latex2htmloptions} module-xml.tex

# Rules for building module-latex docs

module-latex-html : module-latex/index.html

module-latex-print: module-latex.pdf 

module-latex.dvi: module-latex.tex
	-${latex} $<
	-${latex} $<

module-latex.pdf: module-latex.dvi
	dvipdfm -p letter ${basename $<}

module-latex/index.html: module-latex.dvi
	latex2html ${latex2htmloptions} module-latex.tex

# Cleanage

clean:
	-${rmfiles} *.aux *.dvi *.lot *.toc *.log *~
	-${rmfiles} *.[ph]xml documentum.tex *.out

veryclean: clean
	-${rmfiles} *.html *.pdf
	-${rmdirs} module-xml module-latex