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

parms_chunk=chunk.section.depth=2 use.id.as.filename=1
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
rxmlsrcs := ${wildcard *.rxml */*.rxml}
psfigures := \
  ${patsubst %.gif,%.ps,${wildcard */*.gif}} \
  ${patsubst %.jpg,%.ps,${wildcard */*.jpg}}

edition := ../../edition.xml

.PHONY: all html print clean veryclean

.DELETE_ON_ERROR:

all: html print

# HTML versions
html: nets.html

nets.html: ${rxmlsrcs} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=html eruby --delimiters xml $$f > $${f%.rxml}.hxml ; \
	done
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_nochunk} >$@
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_chunk} ${parms_chunk}

# Print version
print: nets.pdf

nets.pdf : ${rxmlsrcs} ${psfigures} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=print eruby --delimiters xml $$f > $${f%.rxml}.pxml ; \
	done
	openjade -t tex -V tex-backend -c ${CATALOG} -o nets.tex \
          -d ${stylesheet_print} ${params_print} ${XML_DCL} main.pxml
	for i in 1 2 ; do jadetex nets.tex ; done
	dvipdfm -p letter nets.dvi


# Image conversion rules.
%.ps : %.gif
	convert -geometry '400x600>' $< $@

%.ps : %.jpg
	convert -geometry '400x600>' $< $@


# Cleanage.
clean:
	-rm nets.aux nets.log nets.out nets.dvi nets.tex */*.ps

veryclean: clean
	-rm *.html nets.pdf *.[hp]xml */*.[hp]xml
