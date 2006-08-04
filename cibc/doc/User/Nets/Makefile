#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


parms_chunk=chunk.section.depth=2 use.id.as.filename=1
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
rxmlsrcs := ${wildcard *.rxml */*.rxml}
psfigures := \
  ${patsubst %.gif,%.ps,${wildcard */*.gif}} \
  ${patsubst %.jpg,%.ps,${wildcard */*.jpg}}
treetop := ../../..
edition := ${treetop}/src/main/sci_version.h

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
