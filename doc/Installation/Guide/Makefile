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

# This Makefile uses the Saxon XSLT processor.  CLASSPATH must point to
# saxon.jar

# STYLESHEET_PATH must point to the location of the srdocbook stylesheets,
#  e.g. /Applications/docbook-xsl-1.51.1/html/.

PARAMS_COMMON=
PARMS_CHUNK=${PARAMS_COMMON} chunk.section.depth=1 use.id.as.filename=1
PARAMS_NOCHUNK=${PARAMS_COMMON} 
STYLESHEET_CHUNK=${STYLESHEET_PATH}/srdocbook-chunk.xsl
STYLESHEET_NOCHUNK=${STYLESHEET_PATH}/srdocbook.xsl
SRCS=Makefile main.xml cvs.xml intro.xml \
recommendations.xml requirements.xml \
rpm_inst.xml source_inst.xml petsc.xml  \
preparing.xml starting.xml help.xml

.PHONY: default chunk nochunk clean veryclean

default: chunk nochunk

chunk: index.html

main.xml: main.rxml ../../edition.xml
	eruby --delimiters xml main.rxml > $@

index.html: ${SRCS}
	-java com.icl.saxon.StyleSheet -w2 \
	main.xml ${STYLESHEET_CHUNK} ${PARMS_CHUNK}

nochunk: Guide.html

.DELETE_ON_ERROR: Guide.html
Guide.html: ${SRCS}
	java com.icl.saxon.StyleSheet -w2 \
	main.xml ${STYLESHEET_NOCHUNK} ${PARAMS_NOCHUNK} >$@

clean:
	-rm -f *~

veryclean: clean
	-rm -f main.xml *.html