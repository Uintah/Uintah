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

PARAMS=
edition:=../../edition.xml
stylesheet_html := ${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print := ${STYLESHEET_DSSSL_PRINT}/docbook.dsl

.PHONY: clean veryclean

.DELETE_ON_ERROR:

all: html pdf

bn := faq

html: ${bn}.html

pdf: ${bn}.pdf

${bn}.html: ${bn}.html.xml
	java com.icl.saxon.StyleSheet -w2 \
	${bn}.html.xml ${stylesheet_html} >$@

${bn}.tex: ${bn}.print.xml
	openjade -t tex -V tex-backend -c ${CATALOG} -o ${bn}.tex \
-d ${stylesheet_print} ${XML_DCL} ${bn}.print.xml 

${bn}.dvi: ${bn}.tex
	for i in 1 2 ; do jadetex ${bn}.tex ; done

${bn}.pdf: ${bn}.dvi
	dvipdfm -p letter ${bn}.dvi

${bn}.html.xml: ${bn}.rxml ${edition}
	OUTPUT_MODE=html eruby --delimiters xml ${bn}.rxml > $@

${bn}.print.xml: ${bn}.rxml ${edition}
	OUTPUT_MODE=print eruby --delimiters xml ${bn}.rxml > $@

clean:
	-rm -f *~

veryclean: clean
	-rm -f ${bn}.*.xml ${bn}.html ${bn}.tex ${bn}.dvi ${bn}.pdf ${bn}.aux \
${bn}.log ${bn}.out

