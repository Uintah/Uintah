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
# --
#
# Make html and ps versions of the Users guide.

# ########## Variables

# edition file
edition := ../../edition.xml

# Latex command
latex := latex -interaction=nonstopmode

# Name of root doc sans extension.
bn := usersguide

# The macros file used by latex and latex2html is generated from this
# one.
rmacrosfile := srugmacros.rtex

# Macros used by latex and latex2html.  Generated from ${rmacrosfile}
macrosfile := srugmacros.tex

# Style file used by latex.   This file has the exact same content as
# ${macrosfile} and is automatically generated from it.
styfile := srugmacros.sty

# Figures
psfigs := Figures/*.gz 
htmlfigs := Figures/*.gif Figures/*.jpg

# User created/edited sources (generated sources come into play further
# down). 
srcs := ${macrosfile} usersguide.tex title.tex intro.tex concepts.tex \
running.tex network.tex viewer.tex packages.tex converters.tex \
itk_sr.tex ../../Utilities/TeX/scirun-doc.tex

# Latex2html options.  Change as needed.
latex2htmloptions = -split 4 -no_white -link 1 -bottom_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix srug -image_type gif \
-style ../../../Utilities/HTML/srlatex2html.css

mainfile := ${bn}.tex
dvifile := ${bn}.dvi
pdffile := ${bn}.pdf
htmldir := ${bn}
treetop := ../../..

# Remove list of directories
define rmdirs
${if ${1},${shell rm -rf ${1}}}
endef

# Remove list of files
define rmfiles
${if ${1},${shell rm -f ${1}}}
endef

# ########## Targets for making various forms of the users guide.

.PHONY: all html dvi pdf clean veryclean

all: dvi pdf html

dvi: ${dvifile}

html: ${htmldir}

pdf: ${pdffile}

${dvifile}: ${srcs} ${psfigs}
	-${latex}  ${bn}
	-makeindex ${bn}.idx
	-bibtex ${bn}
	-${latex} ${bn}
	-${latex} ${bn}

${macrosfile}: ${rmacrosfile} ${edition}
	eruby --delimiters tex $< >$@
	cp ${macrosfile} ${styfile}

${htmldir}: ${dvifile} ${htmlfigs}
	${call rmdirs,${htmldir}}
	latex2html ${latex2htmloptions} ${mainfile}

${pdffile}: ${dvifile}
	dvipdfm -o ${pdffile} -p letter ${dvifile}

# ########## Cleanage
clean:
	${call rmfiles,${bn}.aux ${bn}.log ${bn}.toc ${bn}.lot ${bn}.lof ${bn}.dvi ${bn}.blg ${bn}.idx ${bn}.ilg  ${bn}.ind}

veryclean: clean
	${call rmdirs,${htmldir}}
	${call rmfiles,${pdffile} ${bn}.bbl ${macrosfile} ${styfile}}
