#
# Makefile template for latex docs. Makes dvi, html, 
# and ps versions of a latex doc.
#
# Targets:
#  all         : Make dvi, html, and ps forms of document.
#  dvi         : Make a dvi file.
#  html        : Make html document.
#  ps          : Make a ps document.
#  clean       : Remove aux, log, toc, lot, lof files
#  veryclean   : Remove all of the above plus the html directory,
#                the dvi file, the ps file and *~ files.
#  installhtml : Installs html directory somewhere.
#  installps   : Installs ps file somewhere.
#

.PHONY: all latex html ps installhtml installps

# Latex command
LATEX=latex -interaction=nonstopmode

# Provide name of root doc without its extension.
BN = usersguide

# Provide name of other sources
# Note: Probably should add figures to this list.
SRCS = srugmacros.tex title.tex intro.tex concepts.tex \
running.tex network.tex viewer.tex import.tex

# Latex2html options.  Change as needed.
LATEX2HTMLOPTIONS = -split 4 -no_white -link 1 -bottom_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix srug -image_type gif \
-style ../../doc_styles.css

# Change to reflect installation locations.
HTMLINSTDIR = 
PSINSTDIR = 

MAINFILE = ${BN}.tex
DVIFILE = ${BN}.dvi
PSFILE = ${BN}.ps
# ${PSFILE2} is a ps file optimized for use with distiller.
PSFILE2 = ${BN}2.ps
GZPSFILE = ${PSFILE}.gz
GZPDFFILE = ${PDFFILE}.gz
HTMLDIR = ${BN}
HTMLFILE = ${HTMLDIR}/${BN}.html

all: dvi html ps bib

dvi: ${DVIFILE}

html: ${HTMLFILE}

ps: ${PSFILE}

bib: ${BN}.bbl

${DVIFILE}: ${MAINFILE} ${SRCS} ${BN}.bbl
	${LATEX}  ${BN}
	makeindex ${BN}.idx
	${LATEX} ${BN}

#	for i in 1 2 3; do ${LATEX} ${MAINFILE}; done

${BN}.bbl: ${MAINFILE} 
	if [ ! -r ${BN}.aux ]; \
	then \
	${LATEX} ${BN}; \
	fi; 
	rm -f ${BN}.bbl
	bibtex ${BN}

${HTMLFILE}: ${DVIFILE}
	latex2html ${LATEX2HTMLOPTIONS} ${MAINFILE}

${PSFILE}: ${DVIFILE}
	dvips -o ${PSFILE} -Pps ${DVIFILE}
	cat ${PSFILE} | gzip -c > ${GZPSFILE}
	dvips -o ${PSFILE2} -Ppdf ${DVIFILE}

clean:
	-rm -f ${BN}.aux ${BN}.log ${BN}.toc ${BN}.lot ${BN}.lof

veryclean: clean
	-rm -rf ${HTMLDIR}
	-rm ${BN}.dvi ${BN}.ps ${BN}.pdf ${BN}.*.gz *~

installhtml: ${HTMLFILE}
	if [ -d ${HTMLINSTDIR} ]; \
	then \
	  rm -rf ${HTMLINSTDIR}; \
	fi; \
	cp -r ${HTMLDIR} ${HTMLINSTDIR}

installps: ${PSFILE}
	cp ${PSFILE} ${PSINSTDIR}