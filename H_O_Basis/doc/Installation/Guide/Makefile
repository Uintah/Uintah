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


# Make html and print versions of the installation guide.

parms_chunk := chunk.section.depth=1 use.id.as.filename=1
stylesheet_chunk := ${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk := ${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print := ${STYLESHEET_DSSSL_PRINT}/docbook.dsl
params_print := -V %footnote-ulinks% -V %section-autolabel% \
  -V %generate-article-toc% -V bop-footnotes
rxmlsrcs := ${wildcard *.rxml}
edition := ../../edition.xml
treetop := ../../..

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
          main.hxml ${stylesheet_nochunk} treetop=${treetop} >$@
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_chunk} ${parms_chunk} treetop=${treetop}

# Print version
print: srig.pdf

srig.pdf : ${rxmlsrcs} ${edition}
	for f in ${rxmlsrcs}; \
	do \
	   OUTPUT_MODE=print eruby --delimiters xml $$f > $${f%.rxml}.pxml ; \
	done
	openjade -t tex -V tex-backend -c ${CATALOG} -o srig.tex \
          -d ${stylesheet_print} ${params_print} ${XML_DCL} main.pxml
	for i in 1 2 3; do jadetex srig.tex ; done
	dvipdfm -p letter srig.dvi

# Cleanage
clean:
	-rm -f *.[ph]xml srig.aux srig.log srig.out srig.dvi srig.tex *~

veryclean: clean
	-rm -f *.html srig.pdf
