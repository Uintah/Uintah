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
# The following env variables must be set:
#
# CLASSPATH: Points to .../saxon.jar.
# STYLESHEET_XSL_HTML: Points at .../html
# STYLESHEET_DSSSL_PRINT: Points at .../print
# XML_DCL: Points at .../xml.dcl
# CATALOG: Points at the catalog file that in turn points to
#  .../openjade/dsssl/catalog and .../ISOEnts/isoent.cat (Ugh!).
#

params_common=
parms_chunk=${params_common} chunk.section.depth=1 use.id.as.filename=1
params_nochunk=${params_common} 
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
srcs=Makefile \
 cvs.xml \
 intro.xml \
 recommendations.xml \
 requirements.xml \
 rpm_inst.xml \
 source_inst.xml \
 petsc.xml  \
 preparing.xml \
 starting.xml \
 help.xml
htmlsrcs:=main.html.xml ${srcs}
printsrcs:=main.print.xml ${srcs}

.PHONY: default chunk nochunk pdf clean veryclean

.DELETE_ON_ERROR:

bn := srig

default: chunk nochunk pdf

chunk: index.html

main.html.xml: main.rxml ../../edition.xml
	OUTPUT_MODE=html eruby --delimiters xml main.rxml > $@

main.print.xml: main.rxml ../../edition.xml
	OUTPUT_MODE=print eruby --delimiters xml main.rxml > $@

index.html: ${htmlsrcs}
	-java com.icl.saxon.StyleSheet -w2 \
	main.html.xml ${stylesheet_chunk} ${parms_chunk}

nochunk: ${bn}.html

${bn}.html: ${htmlsrcs}
	java com.icl.saxon.StyleSheet -w2 \
	main.html.xml ${stylesheet_nochunk} ${params_nochunk} >$@

pdf: ${bn}.pdf

${bn}.pdf : ${bn}.dvi
	dvipdfm -p letter ${bn}.dvi

${bn}.dvi : ${bn}.tex
	for i in 1 2 ; do jadetex ${bn}.tex ; done

${bn}.tex : ${printsrcs}
	openjade -t tex -V tex-backend -c ${CATALOG} -o ${bn}.tex \
-d ${stylesheet_print} ${XML_DCL} main.print.xml

clean:
	-rm -f ${bn}.aux ${bn}.log ${bn}.out ${bn}.dvi *~

veryclean: clean
	-rm -f main.*.xml *.html ${bn}.tex ${bn}.pdf
