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
# Build module html files from their xml and tex equivalents.  
# Also generate a cover page for each package of module descriptions.
#
# Generated html is stored either in an appropriate XML or TeX directory.
#
# The env variable CLASSPATH must point to the location of saxon.jar.
#
# The env variable SRGROUP is expected to be set to one of biopse, scirun, or
#  unitah. If its not set or is empty then biopse is assumed.  Note that
#  uintah is not really supported yet.
#

# ########## Macros (Used with ${call ...} function)

#  Convert xml file to html

# Convert the xml file given by $< to the html file given by $@
define xmltohtml
-java com.icl.saxon.StyleSheet -w2 $< ${stylesheet} treetop=${1} > $@
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

# ########## Variables

# Top of tree relative to this directory
treetop:=../../..

# Location of stylesheet relative to this directory.
stylesheet:=${treetop}/doc/Utilities/XML/component.xsl

# Top of tree relative to SCIRun XML dir
xmlsrtreetop:=../../../

# Top of tree relative to Package XML dirs
xmlpkgtreetop:= ../../../../../

# Edition file
edition:=../../edition.xml

# Script that generates package index files
pkgindexgenscript:=${treetop}/doc/Utilities/HTML/makepkgindex.rb
pkgbycatgenscript:=${treetop}/doc/Utilities/HTML/mkpkgidxbycat.rb

# Phony targets.
.PHONY: default SCIRun BioPSE Teem Fusion VDT MatlabInterface clean veryclean

.DELETE_ON_ERROR:

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

default: index.html ${packages}

index.html : index.rhtml ${edition} 
	eruby --delimiters html $< > $@

# ########## SCIRun
SCIRun_xml_dir:=${call xmldir}
SCIRun_xml_html_files:=${call xmlhtmlfiles,${SCIRun_xml_dir}}
SCIRun_tex_dir:=${call texdir}
SCIRun_tex_html_dirs:=${call texhtmldirs,${SCIRun_tex_dir}}

SCIRun : SCIRun.html

SCIRun.html : ${edition} ${SCIRun_xml_html_files} ${SCIRun_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,SCIRun,${SCIRun_xml_dir},${SCIRun_tex_dir}}

${SCIRun_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlsrtreetop}}

${SCIRun_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

SCIRun_clean:
	@echo $@

SCIRun_veryclean:
	${call rmfiles,${SCIRun_xml_html_files}}
	${call rmfiles,${SCIRun_tex_html_files}}
	${call rmdirs,${SCIRun_tex_html_dirs}}

# ########## BioPSE
BioPSE_xml_dir:=${call xmldir,BioPSE}
BioPSE_xml_html_files:=${call xmlhtmlfiles,${BioPSE_xml_dir}}
BioPSE_tex_dir:=${call texdir,BioPSE}
BioPSE_tex_html_dirs:=${call texhtmldirs,${BioPSE_tex_dir}}

BioPSE : BioPSE.html

BioPSE.html : ${edition} ${BioPSE_xml_html_files} ${BioPSE_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,BioPSE,${BioPSE_xml_dir},${BioPSE_tex_dir}}

${BioPSE_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${BioPSE_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

BioPSE_clean:
	@echo $@

BioPSE_veryclean: BioPSE_clean
	${call rmfiles,${BioPSE_xml_html_files}}
	${call rmfiles,${BioPSE_tex_html_files}}
	${call rmdirs,${BioPSE_tex_html_dirs}}

# ########## Teem
Teem_xml_dir:=${call xmldir,Teem}
Teem_xml_html_files:=${call xmlhtmlfiles,${Teem_xml_dir}}
Teem_tex_dir:=${call texdir,Teem}
Teem_tex_html_dirs:=${call texhtmldirs,${Teem_tex_dir}}

Teem : Teem.html

Teem.html : ${edition} ${Teem_xml_html_files} ${Teem_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,Teem,${Teem_xml_dir},${Teem_tex_dir}}

${Teem_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${Teem_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Teem_clean:
	@echo $@

Teem_veryclean:
	${call rmfiles,${Teem_xml_html_files}}
	${call rmfiles,${Teem_tex_html_files}}
	${call rmdirs,${Teem_tex_html_dirs}}

# ########## VDT
VDT_xml_dir:=${call xmldir,VDT}
VDT_xml_html_files:=${call xmlhtmlfiles,${VDT_xml_dir}}
VDT_tex_dir:=${call texdir,VDT}
VDT_tex_html_dirs:=${call texhtmldirs,${VDT_tex_dir}}

VDT : VDT.html

VDT.html : ${edition} ${VDT_xml_html_files} ${VDT_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,VDT,${VDT_xml_dir},${VDT_tex_dir}}

${VDT_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${VDT_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

VDT_clean:
	@echo $@

VDT_veryclean:
	${call rmfiles,${VDT_xml_html_files}}
	${call rmfiles,${VDT_tex_html_files}}
	${call rmdirs,${VDT_tex_html_dirs}}

# ########## MatlabInterface
MatlabInterface_xml_dir:=${call xmldir,MatlabInterface}
MatlabInterface_xml_html_files:=${call xmlhtmlfiles,${MatlabInterface_xml_dir}}
MatlabInterface_tex_dir:=${call texdir,MatlabInterface}
MatlabInterface_tex_html_dirs:=${call texhtmldirs,${MatlabInterface_tex_dir}}

MatlabInterface : MatlabInterface.html

MatlabInterface.html : ${edition} ${MatlabInterface_xml_html_files} \
		${MatlabInterface_tex_html_dirs} ${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,MatlabInterface,${MatlabInterface_xml_dir},${MatlabInterface_tex_dir}}

${MatlabInterface_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${MatlabInterface_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

MatlabInterface_clean:
	@echo $@

MatlabInterface_veryclean: MatlabInterface_clean
	${call rmfiles,${MatlabInterface_xml_html_files}}
	${call rmfiles,${MatlabInterface_tex_html_files}}
	${call rmdirs,${MatlabInterface_tex_html_dirs}}

# ########## Fusion
Fusion_xml_dir:=${call xmldir,Fusion}
Fusion_xml_html_files:=${call xmlhtmlfiles,${Fusion_xml_dir}}
Fusion_tex_dir:=${call texdir,Fusion}
Fusion_tex_html_dirs:=${call texhtmldirs,${Fusion_tex_dir}}

Fusion : Fusion.html

Fusion.html : ${edition} ${Fusion_xml_html_files} ${Fusion_tex_html_dirs} \
		${pkgindexgenscript} ${pkgbycatgenscript}
	${call makepkgindex,Fusion,${Fusion_xml_dir},${Fusion_tex_dir}}

${Fusion_xml_html_files}: %.html: %.xml ${stylesheet}
	${call xmltohtml,${xmlpkgtreetop}}

${Fusion_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Fusion_clean:
	@echo $@

Fusion_veryclean:
	${call rmfiles,${Fusion_xml_html_files}}
	${call rmfiles,${Fusion_tex_html_files}}
	${call rmdirs,${Fusion_tex_html_dirs}}

# ########## Cleanage
clean: ${addsuffix _clean,${packages}}
	@echo clean

veryclean: clean ${addsuffix _veryclean,${packages}}
	${call rmfiles,${addsuffix .html,${packages}} index.html}

