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

# --
#
# Make html and ps versions of the Users guide.

# ########## Variables

treetop := ../../..

# edition file
edition := ${treetop}/src/main/sci_version.h

# Latex command
latex := latex -interaction=nonstopmode

# Latex2html command
latex2html := ${treetop}/doc/Utilities/TeX/latex2html

# Name of root doc sans extension.
bn := usersguide

mainfile := ${bn}.tex
dvifile := ${bn}.dvi
pdffile := ${bn}.pdf
htmldir := ${bn}
texutils := ${treetop}/doc/Utilities/TeX

# Locations of scirun-doc style files
stylefile1 := ${texutils}/scirun-doc.sty
stylefile2 := ${texutils}/scirun-doc.tex
stylefile3 := ${texutils}/scirun-doc.perl

# Symbolic links to style files.
# These style links are a hack to get latex2html to do the right thing.
stylefile1link := scirun-doc.sty
stylefile2link := scirun-doc.tex
stylefile3link := scirun-doc.perl
stylefilelinks := ${stylefile1link} ${stylefile2link} ${stylefile3link}

# The macros file used by latex and latex2html is generated from this
# one.
rmacrosfile := srugmacros.rtex

# Macros used by latex and latex2html.  Generated from ${rmacrosfile}
macrosfile := srugmacros.tex

# Figures
psfigs := Figures/*.gz 
htmlfigs := Figures/*.gif Figures/*.jpg

# User created/edited sources (generated sources come into play further
# down). 
srcs := ${macrosfile} ${mainfile} title.tex intro.tex concepts.tex \
running.tex network.tex viewer.tex packages.tex import_export.tex \
itk_sr.tex ${stylefile1} ${stylefile2} ${stylefile3}

# Latex2html options.  Change as needed.
latex2htmloptions = -split 4 -no_white -link 1 -bottom_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix srug -image_type gif \
-init_file ${treetop}/doc/Utilities/TeX/latex2html-init \
-style ${treetop}/../doc/Utilities/HTML/srlatex2html.css

# Remove list of directories
define rmdirs
${if ${1},${shell rm -rf ${1}}}
endef

# Remove list of files
define rmfiles
${if ${1},${shell rm -f ${1}}}
endef

.PHONY: all html dvi pdf clean veryclean

all: dvi pdf html

dvi: ${dvifile}

html: ${htmldir}

pdf: ${pdffile}

${stylefile1link}:
	ln -s ${stylefile1} ${stylefile1link}

${stylefile2link}:
	ln -s ${stylefile2} ${stylefile2link}

${stylefile3link}:
	ln -s ${stylefile3} ${stylefile3link}

${dvifile}: ${srcs} ${psfigs} ${stylefilelinks}
	-${latex}  ${bn}
	-makeindex ${bn}.idx
	-bibtex ${bn}
	-${latex} ${bn}
	-${latex} ${bn}

${macrosfile}: ${rmacrosfile} ${edition}
	eruby --delimiters tex $< >$@

${htmldir}: ${dvifile} ${htmlfigs}
	${call rmdirs,${htmldir}}
	TREETOP=${treetop}/.. ${latex2html} ${latex2htmloptions} ${mainfile}

${pdffile}: ${dvifile}
	dvipdfm -o ${pdffile} -p letter ${dvifile}

# ########## Cleanage
clean:
	${call rmfiles,${bn}.aux ${bn}.log ${bn}.toc ${bn}.lot ${bn}.lof ${bn}.dvi ${bn}.blg ${bn}.idx ${bn}.ilg  ${bn}.ind}

veryclean: clean
	${call rmdirs,${htmldir}}
	${call rmfiles,${pdffile} ${bn}.bbl ${macrosfile} ${stylefilelinks}}
