# -*- makefile -*-
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
# Build html and print versions of the Module Reference Guide.
#

# ########## Macros (Used with ${call ...} function)

# Convert the xml file given by $< to the html file given by $@
define xmltohtml
-java com.icl.saxon.StyleSheet -w2 $< ${html_stylesheet} treetop=${1} > $@
endef

# Convert tex file given by ${1} to html
define textohtml
ruby ${treetop}/doc/Utilities/TeX/modtex2html.rb ${1}
endef

# Produce list of html files which correspond to xml files in directory
#  given by ${1}
define xmlhtmlfiles
${patsubst %.xml,%.html,${wildcard ${1}/*.xml}}
endef

# Produce list of html directories which correspond to tex files in
#  subdirectories of the directory given by ${1}
define texhtmldirs
${filter-out %/CVS,${foreach n,${notdir ${wildcard ${1}/*}},${1}/${n}/${n}}}
endef


# Return path of XML directory for package given by ${1}
define xmldir
${treetop}/src/${if ${1},Packages/${1}/}Dataflow/XML
endef

# Return path of TeX directory for package given by ${1}
define texdir
${treetop}/src/${if ${1},Packages/${1}/}Dataflow/TeX
endef

# Make package index files.
define makepkgindex
ruby ${pkgindexgenscript} ${1} ${2} ${3}
ruby ${pkgbycatgenscript} ${1} ${2} ${3}
endef 

# Remove list of directories
define rmdirs
${if ${1},${shell rm -rf ${1}}}
endef

# Remove list of files
define rmfiles
${if ${1},${shell rm -f ${1}}}
endef

# Produce list of tex files that correspond to package
#  given by ${1} or to the SCIRun pseudo package if ${1} is empty.
define xmltexfiles
${patsubst %.xml,%.tex,${wildcard ${treetop}/src/${if ${1},Packages/${1}/}Dataflow/XML/*.xml}}
endef

# Produce list of tex source files found in package given by ${1} or in the
# SCIRun pseudo package if ${1} is undefined or empty.
define texfiles
${wildcard ${treetop}/src/${if ${1},Packages/${1}/}Dataflow/TeX/*/*.tex}
endef

# Return path of XML directory for package given by ${1}
# If ${1} is undefined then return XML directory for SCIRun pseudo 
# package.
define xmldir
${treetop}/src/${if ${1},Packages/${1}/}Dataflow/XML
endef

# Convert xml file given by $< to tex equivalent and save tex file as $@.
define xmltotex
-java com.icl.saxon.StyleSheet -w2 $< ${print_stylesheet} frag=1 relpath=${dir $<} > $@
endef

# Make "<Package>_modules.tex" file for package given by ${1}. List of tex
# files is given by ${2}
define makemodtexfile
-test -f ${1}_modules.tex && rm -f ${1}_modules.tex; \
echo "\section{${1} Modules}" >> ${1}_modules.tex; \
for f in ${2}; do echo $$f | ruby ${treetop}/doc/Utilities/TeX/mkpkgmodtex.rb >> ${1}_modules.tex; done
endef

# Return list of eps files that correspond to the image files given by ${1}
# found in doc and TeX directories of the package given by ${2} or the SCIRun
# pseudo package if ${2} is empty.
define imageepsfiles
${patsubst %.${1},%.${eps},${wildcard ${treetop}/src/${if ${2},Packages/${2}/}Dataflow/Modules/*/doc/*.${1}} ${wildcard ${treetop}/src/${if ${2},Packages/${2}/}Dataflow/TeX/*/*.${1}}}
endef

# Convert $< to eps scaling them down if they are too large.
define imagetoeps
-convert -geometry '490x634>' $< EPS:$@
endef

# ########## Variables

# Top of tree relative to this directory
treetop:=../../..

# Location of stylesheet relative to this directory.
html_stylesheet:=${treetop}/doc/Utilities/XML/component.xsl

# Top of tree relative to SCIRun XML dir
xmlsrtreetop:=../../../

# Top of tree relative to Package XML dirs
xmlpkgtreetop:= ../../../../../

# Edition file
edition:=../../edition.xml

# Script that generates package index files
pkgindexgenscript:=${treetop}/doc/Utilities/HTML/makepkgindex.rb
pkgbycatgenscript:=${treetop}/doc/Utilities/HTML/mkpkgidxbycat.rb

# Latex command
latex:=latex -interaction=nonstopmode

# Name of root doc sans extension.
bn := srmrg

# The macros template file used to generate the macros file used by latex.
# one.
rmacrosfile := ${bn}macros.rtex

# Macros used by latex.  Generated from ${rmacrosfile}
macrosfile := ${bn}macros.tex

# Style file used by latex.   This file has the exact same content as
# ${macrosfile} and is automatically generated from it.
styfile := ${bn}macros.sty

# User created/edited sources (generated sources come into play further
# down). 
srcs:=${styfile} ${bn}.tex title.tex modules.tex \
../../Utilities/TeX/scirun-doc.tex

mainfile := ${bn}.tex
dvifile := ${bn}.dvi
pdffile := ${bn}.pdf
print_stylesheet:=${treetop}/doc/Utilities/XML/comp2latex.xsl
eps:=eps

# Docs for a package have been made when ${packagetargets} are fulfilled.
packagetargets := ${addsuffix _modules.tex,${packages}}

# The SRGROUP variable is used to select a group of documentation
# packages.
ifndef SRGROUP
SRGROUP := biopse
endif
ifeq (${strip ${SRGROUP}},)
SRGROUP := biopse
endif

# Set list of packages appropriately.
ifeq (${SRGROUP},scirun)
packages := SCIRun
endif

ifeq (${SRGROUP},biopse)
packages := SCIRun BioPSE Teem MatlabInterface Insight Fusion DataIO
endif

ifeq (${SRGROUP},uintah)
packages := SCIRun Uintah
endif

ifndef SRGROUP
packages :=
endif
ifeq (${strip ${SRGROUP}},)
packages :=
endif

# ########## Primary targets

.PHONY: all dvi pdf
.PHONY: SCIRun BioPSE Teem Fusion MatlabInterface clean veryclean

.DELETE_ON_ERROR:


all: index.html ${packages} dvi pdf

index.html : index.rhtml ${edition} 
	eruby --delimiters html $< > $@

dvi: ${dvifile}

pdf: ${pdffile}

${dvifile}: ${macrosfile} ${mainfile} ${srcs} \
  ${packagetargets}
	-${latex}  ${bn}
	-makeindex ${bn}.idx
	-${latex} ${bn}

${macrosfile}: ${rmacrosfile} ${edition}
	eruby --delimiters tex $< >$@
	cp ${macrosfile} ${styfile}

${pdffile}: ${dvifile}
	dvipdfm -o ${pdffile} -p letter ${dvifile}

# ########## SCIRun
SCIRun : SCIRun.html SCIRun_modules.tex

SCIRun_xml_dir:=${call xmldir}
SCIRun_xml_html_files:=${call xmlhtmlfiles,${SCIRun_xml_dir}}
SCIRun_tex_dir:=${call texdir}
SCIRun_tex_html_dirs:=${call texhtmldirs,${SCIRun_tex_dir}}

SCIRun.html : ${edition} ${SCIRun_xml_html_files} ${SCIRun_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,SCIRun,${SCIRun_xml_dir},${SCIRun_tex_dir}}

${SCIRun_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlsrtreetop}}

${SCIRun_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

SCIRun_xml_tex_files := ${call xmltexfiles}
SCIRun_tex_files := ${call texfiles}
SCIRun_jpg_eps_files := ${call imageepsfiles,jpg}
SCIRun_gif_eps_files := ${call imageepsfiles,gif}

SCIRun_modules.tex : ${SCIRun_jpg_eps_files} ${SCIRun_gif_eps_files} \
${SCIRun_xml_tex_files} ${SCIRun_tex_files}
	${call makemodtexfile,SCIRun,${SCIRun_xml_tex_files} ${SCIRun_tex_files}}

${SCIRun_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${SCIRun_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${SCIRun_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

SCIRun_veryclean_eps_files:
	${call rmfiles,${SCIRun_jpg_eps_files}}
	${call rmfiles,${SCIRun_gif_eps_files}}

SCIRun_veryclean_xml_tex_files:
	${call rmfiles,${SCIRun_xml_tex_files} SCIRun_modules.tex}

SCIRun_clean:
	@echo $@

SCIRun_veryclean: SCIRun_clean SCIRun_veryclean_eps_files \
SCIRun_veryclean_xml_tex_files
	${call rmfiles,${SCIRun_xml_html_files}}
	${call rmfiles,${SCIRun_tex_html_files}}
	${call rmdirs,${SCIRun_tex_html_dirs}}

# ########## BioPSE
BioPSE : BioPSE.html BioPSE_modules.tex

BioPSE_xml_dir:=${call xmldir,BioPSE}
BioPSE_xml_html_files:=${call xmlhtmlfiles,${BioPSE_xml_dir}}
BioPSE_tex_dir:=${call texdir,BioPSE}
BioPSE_tex_html_dirs:=${call texhtmldirs,${BioPSE_tex_dir}}

BioPSE.html : ${edition} ${BioPSE_xml_html_files} ${BioPSE_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,BioPSE,${BioPSE_xml_dir},${BioPSE_tex_dir}}

${BioPSE_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${BioPSE_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

BioPSE_xml_tex_files := ${call xmltexfiles,BioPSE}
BioPSE_jpg_eps_files := ${call imageepsfiles,jpg,BioPSE}
BioPSE_gif_eps_files := ${call imageepsfiles,gif,BioPSE}
BioPSE_tex_files := ${call texfiles,BioPSE}

BioPSE_modules.tex : ${BioPSE_jpg_eps_files} ${BioPSE_gif_eps_files} \
${BioPSE_xml_tex_files} ${BioPSE_tex_files}
	${call makemodtexfile,BioPSE,${BioPSE_xml_tex_files} ${BioPSE_tex_files}}

${BioPSE_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${BioPSE_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${BioPSE_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

BioPSE_veryclean_eps_files:
	${call rmfiles,${BioPSE_jpg_eps_files}}
	${call rmfiles,${BioPSE_gif_eps_files}}

BioPSE_veryclean_xml_tex_files:
	${call rmfiles,${BioPSE_xml_tex_files} BioPSE_modules.tex}

BioPSE_clean:
	@echo $@

BioPSE_veryclean: BioPSE_clean BioPSE_veryclean_eps_files BioPSE_veryclean_xml_tex_files
	${call rmfiles,${BioPSE_xml_html_files}}
	${call rmfiles,${BioPSE_tex_html_files}}
	${call rmdirs,${BioPSE_tex_html_dirs}}


# ########## Teem
Teem : Teem.html Teem_modules.tex

Teem_xml_dir:=${call xmldir,Teem}
Teem_xml_html_files:=${call xmlhtmlfiles,${Teem_xml_dir}}
Teem_tex_dir:=${call texdir,Teem}
Teem_tex_html_dirs:=${call texhtmldirs,${Teem_tex_dir}}

Teem.html : ${edition} ${Teem_xml_html_files} ${Teem_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,Teem,${Teem_xml_dir},${Teem_tex_dir}}

${Teem_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${Teem_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Teem_xml_tex_files := ${call xmltexfiles,Teem}
Teem_jpg_eps_files := ${call imageepsfiles,jpg,Teem}
Teem_gif_eps_files := ${call imageepsfiles,gif,Teem}
Teem_tex_files := ${call texfiles,Teem}

Teem_modules.tex : ${Teem_jpg_eps_files} ${Teem_gif_eps_files} \
${Teem_xml_tex_files} ${Teem_tex_files}
	${call makemodtexfile,Teem,${Teem_xml_tex_files} ${Teem_tex_files}}

${Teem_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${Teem_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${Teem_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

Teem_veryclean_eps_files:
	${call rmfiles,${Teem_jpg_eps_files}}
	${call rmfiles,${Teem_gif_eps_files}}

Teem_veryclean_xml_tex_files:
	${call rmfiles,${Teem_xml_tex_files} Teem_modules.tex}

Teem_clean:
	@echo $@

Teem_veryclean: Teem_clean Teem_veryclean_eps_files \
Teem_veryclean_xml_tex_files
	${call rmfiles,${Teem_xml_html_files}}
	${call rmfiles,${Teem_tex_html_files}}
	${call rmdirs,${Teem_tex_html_dirs}}

# ########## MatlabInterface
MatlabInterface : MatlabInterface.html MatlabInterface_modules.tex

MatlabInterface_xml_dir:=${call xmldir,MatlabInterface}
MatlabInterface_xml_html_files:=${call xmlhtmlfiles,${MatlabInterface_xml_dir}}
MatlabInterface_tex_dir:=${call texdir,MatlabInterface}
MatlabInterface_tex_html_dirs:=${call texhtmldirs,${MatlabInterface_tex_dir}}

MatlabInterface.html : ${edition} ${MatlabInterface_xml_html_files} ${MatlabInterface_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,MatlabInterface,${MatlabInterface_xml_dir},${MatlabInterface_tex_dir}}

${MatlabInterface_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${MatlabInterface_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

MatlabInterface_xml_tex_files := ${call xmltexfiles,MatlabInterface}
MatlabInterface_jpg_eps_files := ${call imageepsfiles,jpg,MatlabInterface}
MatlabInterface_gif_eps_files := ${call imageepsfiles,gif,MatlabInterface}
MatlabInterface_tex_files := ${call texfiles,MatlabInterface}

MatlabInterface_modules.tex : ${MatlabInterface_jpg_eps_files} ${MatlabInterface_gif_eps_files} \
${MatlabInterface_xml_tex_files} ${MatlabInterface_tex_files}
	${call makemodtexfile,MatlabInterface,${MatlabInterface_xml_tex_files} ${MatlabInterface_tex_files}}

${MatlabInterface_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${MatlabInterface_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${MatlabInterface_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

MatlabInterface_veryclean_eps_files:
	${call rmfiles,${MatlabInterface_jpg_eps_files}}
	${call rmfiles,${MatlabInterface_gif_eps_files}}

MatlabInterface_veryclean_xml_tex_files:
	${call rmfiles,${MatlabInterface_xml_tex_files} MatlabInterface_modules.tex}

MatlabInterface_clean:
	@echo $@

MatlabInterface_veryclean: MatlabInterface_clean \
MatlabInterface_veryclean_eps_files \
MatlabInterface_veryclean_xml_tex_files
	${call rmfiles,${MatlabInterface_xml_html_files}}
	${call rmfiles,${MatlabInterface_tex_html_files}}
	${call rmdirs,${MatlabInterface_tex_html_dirs}}

# ########## Insight
Insight : Insight.html Insight_modules.tex

Insight_xml_dir:=${call xmldir,Insight}
Insight_xml_html_files:=${call xmlhtmlfiles,${Insight_xml_dir}}
Insight_tex_dir:=${call texdir,Insight}
Insight_tex_html_dirs:=${call texhtmldirs,${Insight_tex_dir}}

Insight.html : ${edition} ${Insight_xml_html_files} ${Insight_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,Insight,${Insight_xml_dir},${Insight_tex_dir}}

${Insight_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${Insight_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Insight_xml_tex_files := ${call xmltexfiles,Insight}
Insight_jpg_eps_files := ${call imageepsfiles,jpg,Insight}
Insight_gif_eps_files := ${call imageepsfiles,gif,Insight}
Insight_tex_files := ${call texfiles,Insight}

Insight_modules.tex : ${Insight_jpg_eps_files} ${Insight_gif_eps_files} \
${Insight_xml_tex_files} ${Insight_tex_files}
	${call makemodtexfile,Insight,${Insight_xml_tex_files} ${Insight_tex_files}}

${Insight_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${Insight_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${Insight_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

Insight_veryclean_eps_files:
	${call rmfiles,${Insight_jpg_eps_files}}
	${call rmfiles,${Insight_gif_eps_files}}

Insight_veryclean_xml_tex_files:
	${call rmfiles,${Insight_xml_tex_files} Insight_modules.tex}

Insight_clean:
	@echo $@

Insight_veryclean: Insight_clean \
Insight_veryclean_eps_files \
Insight_veryclean_xml_tex_files
	${call rmfiles,${Insight_xml_html_files}}
	${call rmfiles,${Insight_tex_html_files}}
	${call rmdirs,${Insight_tex_html_dirs}}

# ########## Fusion
Fusion : Fusion.html Fusion_modules.tex

Fusion_xml_dir:=${call xmldir,Fusion}
Fusion_xml_html_files:=${call xmlhtmlfiles,${Fusion_xml_dir}}
Fusion_tex_dir:=${call texdir,Fusion}
Fusion_tex_html_dirs:=${call texhtmldirs,${Fusion_tex_dir}}

Fusion.html : ${edition} ${Fusion_xml_html_files} ${Fusion_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,Fusion,${Fusion_xml_dir},${Fusion_tex_dir}}

${Fusion_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${Fusion_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Fusion_xml_tex_files := ${call xmltexfiles,Fusion}
Fusion_jpg_eps_files := ${call imageepsfiles,jpg,Fusion}
Fusion_gif_eps_files := ${call imageepsfiles,gif,Fusion}
Fusion_tex_files := ${call texfiles}

Fusion_modules.tex : ${Fusion_jpg_eps_files} ${Fusion_gif_eps_files} \
${Fusion_xml_tex_files} ${Fusion_tex_files}
	${call makemodtexfile,Fusion,${Fusion_xml_tex_files} ${Fusion_tex_files}}

${Fusion_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${Fusion_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${Fusion_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

Fusion_veryclean_eps_files:
	${call rmfiles,${Fusion_jpg_eps_files}}
	${call rmfiles,${Fusion_gif_eps_files}}

Fusion_veryclean_xml_tex_files:
	${call rmfiles,${Fusion_xml_tex_files} Fusion_modules.tex}

Fusion_clean:
	@echo $@

Fusion_veryclean: Fusion_clean Fusion_veryclean_eps_files \
Fusion_veryclean_xml_tex_files
	${call rmfiles,${Fusion_xml_html_files}}
	${call rmfiles,${Fusion_tex_html_files}}
	${call rmdirs,${Fusion_tex_html_dirs}}

# ########## DataIO
DataIO : DataIO.html DataIO_modules.tex

DataIO_xml_dir:=${call xmldir,DataIO}
DataIO_xml_html_files:=${call xmlhtmlfiles,${DataIO_xml_dir}}
DataIO_tex_dir:=${call texdir,DataIO}
DataIO_tex_html_dirs:=${call texhtmldirs,${DataIO_tex_dir}}

DataIO.html : ${edition} ${DataIO_xml_html_files} ${DataIO_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,DataIO,${DataIO_xml_dir},${DataIO_tex_dir}}

${DataIO_xml_html_files}: %.html: %.xml ${html_stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${DataIO_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

DataIO_xml_tex_files := ${call xmltexfiles,DataIO}
DataIO_jpg_eps_files := ${call imageepsfiles,jpg,DataIO}
DataIO_gif_eps_files := ${call imageepsfiles,gif,DataIO}
DataIO_tex_files := ${call texfiles}

DataIO_modules.tex : ${DataIO_jpg_eps_files} ${DataIO_gif_eps_files} \
${DataIO_xml_tex_files} ${DataIO_tex_files}
	${call makemodtexfile,DataIO,${DataIO_xml_tex_files} ${DataIO_tex_files}}

${DataIO_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${DataIO_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${DataIO_xml_tex_files}: %.tex : %.xml ${print_stylesheet}
	${call xmltotex}

DataIO_veryclean_eps_files:
	${call rmfiles,${DataIO_jpg_eps_files}}
	${call rmfiles,${DataIO_gif_eps_files}}

DataIO_veryclean_xml_tex_files:
	${call rmfiles,${DataIO_xml_tex_files} DataIO_modules.tex}

DataIO_clean:
	@echo $@

DataIO_veryclean: DataIO_clean DataIO_veryclean_eps_files \
DataIO_veryclean_xml_tex_files
	${call rmfiles,${DataIO_xml_html_files}}
	${call rmfiles,${DataIO_tex_html_files}}
	${call rmdirs,${DataIO_tex_html_dirs}}

# ########## Cleanage
clean: ${addsuffix _clean,${packages}}
	${call rmfiles,${bn}.aux ${bn}.log ${bn}.toc ${bn}.lot \
	${bn}.lof ${bn}.dvi ${bn}.blg ${bn}.idx ${bn}.ilg \
	${bn}.ind}

veryclean: clean ${addsuffix _veryclean,${packages}}
	${call rmfiles,${addsuffix .html,${packages}} ${addsuffix _bycat.html,${packages}} index.html} 
	${call rmfiles,${pdffile} ${macrosfile} ${styfile}}

