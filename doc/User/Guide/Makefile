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
#
# The env variable SRGROUP is expected to be set to one of scirun, biopse, or
#  unitah. If its not set or is empty then biopse is assumed.  Note that
#  uintah is not really supported yet.  This variable selects the set of
#  packages for which module reference documents should be generated.
#

# ########## Variables

# edition file
edition:=../../edition.xml

# Latex command
latex:=latex -interaction=nonstopmode

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
srcs:=${styfile} usersguide.tex title.tex intro.tex concepts.tex \
running.tex network.tex viewer.tex packages.tex converters.tex \
modules.tex ../../Utilities/TeX/scirun-doc.tex

# Latex2html options.  Change as needed.
latex2htmloptions = -split 4 -no_white -link 1 -bottom_navigation \
-html_version 4.0,math -show_section_numbers -local_icons \
-long_titles 1 -prefix srug -image_type gif \
-style ../../../Utilities/HTML/srlatex2html.css

mainfile := ${bn}.tex
dvifile := ${bn}.dvi
pdffile = ${bn}.pdf
htmldir := ${bn}
treetop:=../../..
stylesheet:=${treetop}/doc/Utilities/XML/comp2latex.xsl
eps:=eps

# The SRGROUP variable is used to select a group of documentation
# packages.  See below.

# Check SRGROUP variable...
ifndef SRGROUP
SRGROUP := biopse
endif
ifeq (${strip ${SRGROUP}},)
SRGROUP := biopse
endif

# ...and set list of packages appropriately.

ifeq (${SRGROUP},scirun)
packages := SCIRun
endif

ifeq (${SRGROUP},biopse)
packages := SCIRun BioPSE Teem VDT MatlabInterface Fusion
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

# Docs for a package have been made when then these targets are fulfilled.
packagetargets := ${addsuffix _modules.tex,${packages}}

# ########## Macros (Used with ${call ...} function)

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
-java com.icl.saxon.StyleSheet -w2 $< ${stylesheet} frag=1 relpath=${dir $<} > $@
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

# Remove list of directories
define rmdirs
${if ${1},${shell rm -rf ${1}}}
endef

# Remove list of files
define rmfiles
${if ${1},${shell rm -f ${1}}}
endef

# ########## Targets for making various forms of the users guide.

.PHONY: all latex html pdf bib

all: dvi pdf html  bib

dvi: ${dvifile}

html: ${htmldir}

pdf: ${pdffile}

bib: ${bn}.bbl

# Note: ${bn}.bbl must be the last dependency!
${dvifile}: ${macrosfile} ${mainfile} ${srcs} ${psfigs} \
  ${packagetargets} ${bn}.bbl 
	-${latex}  ${bn}
	-makeindex ${bn}.idx
	-${latex} ${bn}

${macrosfile}: ${rmacrosfile} ${edition}
	eruby --delimiters tex $< >$@
	cp ${macrosfile} ${styfile}

${bn}.bbl: ${mainfile}
	-${latex} ${bn}
	${call rmfiles,${bn}.bbl}
	-bibtex ${bn}

${htmldir}: ${dvifile} ${htmlfigs}
	${call rmdirs,${htmldir}}
	latex2html ${latex2htmloptions} ${mainfile}

${pdffile}: ${dvifile}
	dvipdfm -o ${pdffile} -p letter ${dvifile}

# ########## Targets for making tex version module docs.

.PHONY: SCIRun BioPSE Teem Fusion VDT MatlabInterface clean veryclean

# ########## SCIRun
SCIRun_xml_tex_files := ${call xmltexfiles}
SCIRun_jpg_eps_files := ${call imageepsfiles,jpg}
SCIRun_gif_eps_files := ${call imageepsfiles,gif}
SCIRun_tex_files := ${call texfiles}

SCIRun : SCIRun_modules.tex

SCIRun_modules.tex : ${SCIRun_jpg_eps_files} ${SCIRun_gif_eps_files} \
${SCIRun_xml_tex_files} ${SCIRun_tex_files}
	${call makemodtexfile,SCIRun,${SCIRun_xml_tex_files} ${SCIRun_tex_files}}

${SCIRun_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${SCIRun_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${SCIRun_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

SCIRun_clean:
	@echo $@

SCIRun_veryclean_eps_files:
	${call rmfiles,${SCIRun_jpg_eps_files}}
	${call rmfiles,${SCIRun_gif_eps_files}}

SCIRun_veryclean_xml_tex_files:
	${call rmfiles,${SCIRun_xml_tex_files} SCIRun_modules.tex}

SCIRun_veryclean: SCIRun_clean SCIRun_veryclean_eps_files SCIRun_veryclean_xml_tex_files

# ########## BioPSE
BioPSE_xml_tex_files := ${call xmltexfiles,BioPSE}
BioPSE_jpg_eps_files := ${call imageepsfiles,jpg,BioPSE}
BioPSE_gif_eps_files := ${call imageepsfiles,gif,BioPSE}
BioPSE_tex_files := ${call texfiles,BioPSE}

BioPSE : BioPSE_modules.tex

BioPSE_modules.tex : ${BioPSE_jpg_eps_files} ${BioPSE_gif_eps_files} \
${BioPSE_xml_tex_files} ${BioPSE_tex_files}
	${call makemodtexfile,BioPSE,${BioPSE_xml_tex_files} ${BioPSE_tex_files}}

${BioPSE_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${BioPSE_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${BioPSE_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

BioPSE_clean:
	@echo $@

BioPSE_veryclean_eps_files:
	${call rmfiles,${BioPSE_gif_eps_files}}
	${call rmfiles,${BioPSE_jpg_eps_files}}

BioPSE_veryclean_xml_tex_files:
	${call rmfiles,${BioPSE_xml_tex_files} BioPSE_modules.tex}

BioPSE_veryclean: BioPSE_clean BioPSE_veryclean_eps_files BioPSE_veryclean_xml_tex_files


# ########## Teem
Teem_xml_tex_files := ${call xmltexfiles,Teem}
Teem_jpg_eps_files := ${call imageepsfiles,jpg,Teem}
Teem_gif_eps_files := ${call imageepsfiles,gif,Teem}
Teem_tex_files := ${call texfiles,Teem}


Teem : Teem_modules.tex

Teem_modules.tex : ${Teem_jpg_eps_files} ${Teem_gif_eps_files} \
${Teem_xml_tex_files} ${Teem_tex_files}
	${call makemodtexfile,Teem,${Teem_xml_tex_files} ${Teem_tex_files}}

${Teem_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${Teem_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${Teem_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

Teem_clean:
	@echo $@

Teem_veryclean_eps_files:
	${call rmfiles,${Teem_jpg_eps_files}}
	${call rmfiles,${Teem_gif_eps_files}}

Teem_veryclean_xml_tex_files:
	${call rmfiles,${Teem_xml_tex_files} Teem_modules.tex}

Teem_veryclean: Teem_clean Teem_veryclean_eps_files Teem_veryclean_xml_tex_files

# ########## Fusion
Fusion_xml_tex_files := ${call xmltexfiles,Fusion}
Fusion_jpg_eps_files := ${call imageepsfiles,jpg,Fusion}
Fusion_gif_eps_files := ${call imageepsfiles,gif,Fusion}
Fusion_tex_files := ${call texfiles,Fusion}

Fusion : Fusion_modules.tex

Fusion_modules.tex : ${Fusion_jpg_eps_files} ${Fusion_gif_eps_files} \
${Fusion_xml_tex_files} ${Fusion_tex_files}
	${call makemodtexfile,Fusion,${Fusion_xml_tex_files} ${Fusion_tex_files}}

${Fusion_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${Fusion_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${Fusion_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

Fusion_clean:
	@echo $@

Fusion_veryclean_eps_files:
	${call rmfiles,${Fusion_jpg_eps_files}}
	${call rmfiles,${Fusion_gif_eps_files}}

Fusion_veryclean_xml_tex_files:
	${call rmfiles,${Fusion_xml_tex_files} Fusion_modules.tex}

Fusion_veryclean: Fusion_clean Fusion_veryclean_eps_files Fusion_veryclean_xml_tex_files

# ########## MatlabInterface
MatlabInterface_xml_tex_files := ${call xmltexfiles,MatlabInterface}
MatlabInterface_jpg_eps_files := ${call imageepsfiles,jpg,MatlabInterface}
MatlabInterface_gif_eps_files := ${call imageepsfiles,gif,MatlabInterface}
MatlabInterface_tex_files := ${call texfiles,MatlabInterface}

MatlabInterface : MatlabInterface_modules.tex

MatlabInterface_modules.tex : ${MatlabInterface_jpg_eps_files} \
${MatlabInterface_gif_eps_files} ${MatlabInterface_xml_tex_files} \
${MatlabInterface_tex_files}
	${call makemodtexfile,MatlabInterface,${MatlabInterface_xml_tex_files} ${MatlabInterface_tex_files}}

${MatlabInterface_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${MatlabInterface_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${MatlabInterface_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

MatlabInterface_clean:
	@echo $@

MatlabInterface_veryclean_eps_files:
	${call rmfiles,${MatlabInterface_jpg_eps_files}}
	${call rmfiles,${MatlabInterface_gif_eps_files}}

MatlabInterface_veryclean_xml_tex_files:
	${call rmfiles,${MatlabInterface_xml_tex_files} MatlabInterface_modules.tex}

MatlabInterface_veryclean: MatlabInterface_clean \
MatlabInterface_veryclean_eps_files MatlabInterface_veryclean_xml_tex_files

# ########## VDT
VDT_xml_tex_files := ${call xmltexfiles,VDT}
VDT_jpg_eps_files := ${call imageepsfiles,jpg,VDT}
VDT_gif_eps_files := ${call imageepsfiles,gif,VDT}
VDT_tex_files := ${call texfiles,VDT}

VDT : VDT_modules.tex

VDT_modules.tex : ${VDT_jpg_eps_files} ${VDT_gif_eps_files} \
${VDT_xml_tex_files} ${VDT_tex_files}
	${call makemodtexfile,VDT,${VDT_xml_tex_files} ${VDT_tex_files}}

${VDT_jpg_eps_files}: %.${eps} : %.jpg
	${call imagetoeps}

${VDT_gif_eps_files}: %.${eps} : %.gif
	${call imagetoeps}

${VDT_xml_tex_files}: %.tex : %.xml ${stylesheet}
	${call xmltotex}

VDT_clean:
	@echo $@

VDT_veryclean_eps_files:
	${call rmfiles,${VDT_jpg_eps_files}}
	${call rmfiles,${VDT_gif_eps_files}}

VDT_veryclean_xml_tex_files:
	${call rmfiles,${VDT_xml_tex_files} VDT_modules.tex}

VDT_veryclean: VDT_clean VDT_veryclean_eps_files VDT_veryclean_xml_tex_files

# ########## Cleanage
clean: ${addsuffix _clean,${packages}}
	${call rmfiles,${bn}.aux ${bn}.log ${bn}.toc ${bn}.lot \
	${bn}.lof ${bn}.dvi ${bn}.blg ${bn}.idx ${bn}.ilg \
	${bn}.ind}

veryclean_xml_tex_files: ${addsuffix _veryclean_xml_tex_files,${packages}}

veryclean_eps_files: ${addsuffix _veryclean_eps_files,${packages}}

veryclean: ${addsuffix _veryclean,${packages}}
	${call rmdirs,${htmldir}}
	${call rmfiles,${pdffile} *.bbl ${macrosfile} ${styfile}}
