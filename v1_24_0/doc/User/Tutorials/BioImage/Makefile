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

parms_chunk := chunk.section.depth=2 use.id.as.filename=1
stylesheet_chunk := ${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk := ${STYLESHEET_XSL_HTML}/srdocbook.xsl

processor := dsssl
ifeq (${processor},fop)
stylesheet_print := ${STYLESHEET_XSL_PRINT}/profile-docbook.xsl
params_print := ${params_common} section.autolabel=1 ulink.footnotes=1 \
  section.label.includes.component.label=1 fop.extensions=1 \
  double.sided=1
else
stylesheet_print := ${STYLESHEET_DSSSL_PRINT}/docbook.dsl
params_print := -V %footnote-ulinks% -V %section-autolabel% \
   -V %generate-article-toc% -V bop-footnotes
endif

rxmlsrcs := ${wildcard *.rxml}
treetop := ../../../..
edition := ${treetop}/src/main/sci_version.h
printfigures := ${patsubst %.gif,%.eps,${wildcard Figures/*.gif}} \
              ${patsubst %.jpg,%.eps,${wildcard Figures/*.jpg}}

.PHONY: all html print clean veryclean

.DELETE_ON_ERROR:

all: html print

printfigs: ${printfigures}

# Implicit rule to generate sources for html output
%.hxml : %.rxml
	TREETOP=${treetop} OUTPUT_MODE=html eruby --delimiters xml $< > $@

# Implicit rule to generate sources for print output
%.pxml : %.rxml
	TREETOP=${treetop} OUTPUT_MODE=print eruby --delimiters xml $< > $@

# Make figures used in print output.
Figures/%.eps : Figures/%.jpg
	convert -density 72 -geometry '360x432>' $< EPS:$@
Figures/%.eps : Figures/%.gif
	convert -density 72 -geometry '360x432>' $< EPS:$@

# HTML versions
html: bioimage.html

hxmlsrcs := ${rxmlsrcs:.rxml=.hxml}
bioimage.html: ${hxmlsrcs} ${edition}
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_nochunk} treetop=${treetop} >$@
	java com.icl.saxon.StyleSheet -w2 \
          main.hxml ${stylesheet_chunk} ${parms_chunk} treetop=${treetop}

# Print version
print: bioimage.pdf

pxmlsrcs := ${rxmlsrcs:.rxml=.pxml}
bioimage.pdf : ${pxmlsrcs} ${edition} ${printfigures}
ifeq (${processor},fop)
	java com.icl.saxon.StyleSheet -w2 -o bioimage.fo \
          main.pxml ${stylesheet_print} ${params_print} >$@
	fop -fo bioimage.fo -pdf bioimage.pdf
else
bioimage.pdf : ${pxmlsrcs} ${edition} ${printfigures}
	openjade -t tex -V tex-backend -c ${CATALOG} -o bioimage.tex \
          -d ${stylesheet_print} ${params_print} ${XML_DCL} main.pxml
	for i in 1 2 3; do jadetex bioimage.tex ; done
	dvipdfm -p letter bioimage.dvi
endif

# Cleanage
clean:
	-rm -f *.[ph]xml bioimage.fo bioimage.aux bioimage.log bioimage.out bioimage.dvi bioimage.tex Figures/*.eps *~

veryclean: clean
	-rm -f *.html bioimage.pdf
