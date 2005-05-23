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


chunk_root := documentum.chunk
params_html=use.extensions=1 tablecolumns.extension=1
params_nochunk=${params_html}
params_chunk=chunk.section.depth=1 use.id.as.filename=1 \
  ${params_html} root.filename=${chunk_root}
stylesheet_chunk=${STYLESHEET_XSL_HTML}/srdocbook-chunk.xsl
stylesheet_nochunk=${STYLESHEET_XSL_HTML}/srdocbook.xsl
stylesheet_print=${STYLESHEET_DSSSL_PRINT}/docbook.dsl
params_print := -V %footnote-ulinks% -V %section-autolabel% \
  -V "(define %default-quadding% 'justify)"
treetop := ../..
edition := ${treetop}/src/main/sci_version.h

latex:=latex -interaction=nonstopmode
latex2htmloptions:= -info 0 -split 0 -no_white -link 1 -no_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix module-latex -image_type gif \
-style ../../Utilities/HTML/srlatex2html.css

# Implicit rule to generate sources for html output
%.hxml : %.rxml
	TREETOP=${treetop} OUTPUT_MODE=html eruby --delimiters xml $< > $@

# Implicit rule to generate sources for print output
%.pxml : %.rxml
	TREETOP=${treetop} OUTPUT_MODE=print eruby --delimiters xml $< > $@

rmfiles:=rm -f
rmdirs:=rm -rf

.PHONY: \
  all \
  documentum-html \
  documentum-print \
  mod-spec-ref-html \
  mod-spec-ref-print \
  doc_inventory-html \
  doc_inventory-print \
  clean \
  veryclean

.DELETE_ON_ERROR:

all: \
  index.html \
  documentum-html \
  documentum-print \
  mod-spec-ref-html \
  doc_inventory-html \
  doc_inventory-print
#  mod-spec-ref-print

# Rule for building index.html
index.html: index.rhtml ${edition}
	eruby --delimiters html $< >$@

# Rules for building documentum.
documentum : documentum-html documentum-print

documentum-html : documentum.html

documentum.html : documentum.hxml ${edition}
	java com.icl.saxon.StyleSheet -w2 documentum.hxml \
         ${stylesheet_chunk} ${params_chunk} treetop=${treetop} 
	java com.icl.saxon.StyleSheet -w2 documentum.hxml \
         ${stylesheet_nochunk} ${params_nochunk} treetop=${treetop} >$@

documentum-print : documentum.pdf

documentum.pdf : documentum.pxml ${edition}
	openjade -t tex -V tex-backend -c ${CATALOG} -o documentum.tex \
-d ${stylesheet_print}  ${params_print} ${XML_DCL} documentum.pxml 
	for i in 1 2 ; do jadetex documentum.tex ; done
	dvipdfm -p letter documentum.dvi

# Rules for building component reference.
mod-spec-ref : mod-spec-ref-html mod-spec-ref-print

mod-spec-ref-html : mod-spec-ref.html

mod-spec-ref.html : mod-spec-ref.hxml mod-spec-ref-partI.hxml mod-spec-ref-partII.hxml ${edition}
	${call rxml_to_hxml,mod-spec-ref.rxml}
	${call rxml_to_hxml,mod-spec-ref-partI.rxml}
	${call rxml_to_hxml,mod-spec-ref-partII.rxml}
	java com.icl.saxon.StyleSheet -w2 mod-spec-ref.hxml \
         ${stylesheet_chunk} ${params_chunk} treetop=${treetop} 
	java com.icl.saxon.StyleSheet -w2 mod-spec-ref.hxml \
         ${stylesheet_nochunk} ${params_nochunk} treetop=${treetop} >$@

mod-spec-ref-print : mod-spec-ref.pdf

mod-spec-ref.pdf : mod-spec-ref.pxml mod-spec-ref-partI.pxml mod-spec-ref-partII.pxml ${edition}
	${call rxml_to_pxml,mod-spec-ref.rxml}
	${call rxml_to_pxml,mod-spec-ref-partI.rxml}
	${call rxml_to_pxml,mod-spec-ref-partII.rxml}
	openjade -t tex -V tex-backend -c ${CATALOG} -o mod-spec-ref.tex \
-d ${stylesheet_print}  ${params_print} ${XML_DCL} mod-spec-ref.pxml 
	for i in 1 2 ; do jadetex mod-spec-ref.tex ; done
	dvipdfm -p letter mod-spec-ref.dvi

# Rules for building doc_inventory.

doc_inventory : doc_inventory-html doc_inventory-print

doc_inventory-html : doc_inventory.html

doc_inventory.html : doc_inventory.hxml ${edition}
	java com.icl.saxon.StyleSheet -w2 doc_inventory.hxml \
         ${stylesheet_chunk} ${params_chunk} treetop=${treetop} 
	java com.icl.saxon.StyleSheet -w2 doc_inventory.hxml \
         ${stylesheet_nochunk} ${params_nochunk} treetop=${treetop} >$@

doc_inventory-print : doc_inventory.pdf

doc_inventory.pdf : doc_inventory.pxml ${edition}
	openjade -t tex -V tex-backend -c ${CATALOG} -o doc_inventory.tex \
-d ${stylesheet_print}  ${params_print} ${XML_DCL} doc_inventory.pxml 
	for i in 1 2 ; do jadetex doc_inventory.tex ; done
	dvipdfm -p letter doc_inventory.dvi

# Cleanage

clean:
	-${rmfiles} *.aux *.dvi *.lot *.toc *.log *~
	-${rmfiles} *.[ph]xml documentum.tex mod-spec-ref.tex doc_inventory.tex *.out

veryclean: clean
	-${rmfiles} *.html *.pdf
