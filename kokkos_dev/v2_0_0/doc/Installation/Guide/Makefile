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

# Make html and print versions of the installation guide.

parms_chunk := chunk.section.depth=1 use.id.as.filename=1
stylesheet_chunk := ${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk := ${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print := ${STYLESHEET_DSSSL_PRINT}/docbook.dsl
params_print := -V %footnote-ulinks% -V %section-autolabel% \
  -V "(define %default-quadding% 'justify)"
rxmlsrcs := ${wildcard *.rxml}
edition := ../../edition.xml

.PHONY: all html print clean veryclean

.DELETE_ON_ERROR:

all: html print

# HTML versions
html: srig.html

srig.html: ${rxmlsrcs} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=html eruby --delimiters xml $$f > $${f%.rxml}.hxml ; \
	done
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_nochunk} >$@
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_chunk} ${parms_chunk}

# Print version
print: srig.pdf

srig.pdf : ${rxmlsrcs} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=print eruby --delimiters xml $$f > $${f%.rxml}.pxml ; \
	done
	openjade -t tex -V tex-backend -c ${CATALOG} -o srig.tex \
          -d ${stylesheet_print} ${params_print} ${XML_DCL} main.pxml
	for i in 1 2 ; do jadetex srig.tex ; done
	dvipdfm -p letter srig.dvi

# Cleanage
clean:
	-rm -f *.[ph]xml srig.aux srig.log srig.out srig.dvi srig.tex *~

veryclean: clean
	-rm -f *.html srig.pdf
