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

parms_chunk=chunk.section.depth=0 use.id.as.filename=1
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
rxmlsrcs := ${wildcard *.rxml} ${wildcard dev/*.rxml} ${wildcard src/*.rxml}
psfigures := ${patsubst %.gif,%.ps,${wildcard dev/Figures/*.gif}}
edition := ../../edition.xml

.PHONY: all html print clean veryclean

.DELETE_ON_ERROR:

all: html print

# HTML versions
html: srdg.html

srdg.html: ${rxmlsrcs} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=html eruby --delimiters xml $$f > $${f%.rxml}.hxml ; \
	done
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_nochunk} >$@
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_chunk} ${parms_chunk}

# Print version
print: srdg.pdf

srdg.pdf : ${rxmlsrcs} ${psfigures} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=print eruby --delimiters xml $$f > $${f%.rxml}.pxml ; \
	done
	openjade -t tex -V tex-backend -c ${CATALOG} -o srdg.tex \
          -d ${stylesheet_print} ${params_print} ${XML_DCL} main.pxml
	for i in 1 2 ; do jadetex srdg.tex ; done
	dvipdfm -p letter srdg.dvi

%.ps : %.gif
	convert $< $@

clean:
	-rm -f srdg.aux srdg.log srdg.out srdg.dvi  dev/Figures/*.ps *~

veryclean: clean
	-rm -f dev.*html src.*html *.[ph]xml dev/*.[ph]xml src/*.[ph]xml \
           index.html intro.html srdg.html srdg.tex srdg.pdf
