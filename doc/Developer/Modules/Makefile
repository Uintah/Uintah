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
# The variable SRGROUP is expected to be set to one of biopse, scirun, or
#  unitah. If its not set or is empty then biopse is assumed.
#

# ########## Macros (Used with ${call ...} function)

#  Convert xml file to html
#
# Use saxon 6.5 or higher to process the xml files.
# The environment variable CLASSPATH must point to saxon.jar.
#
define xmltohtml
@echo $@
@-sed "s#\".*component.dtd#\"${treetop}/doc/Utilities/XML/component.dtd#" $< >${notdir $<}
@-java com.icl.saxon.StyleSheet -w2 ${notdir $<} ${stylesheet} \
treetop=${1} > $@
@-rm ${notdir $<}
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

# Produce list of tex files in subdirectories of the directory
#  given by ${1}
define texfiles
${wildcard ${1}/*/*.tex}
endef

# Produce list of html directories which correspond to tex files in
#  subdirectories of the directory given by ${1}
define texhtmldirs
${foreach n,${notdir ${wildcard ${1}/*}},${1}/${n}/${n}}
endef

# Return path of XML directory for package given by ${1}
define xmldir
${treetop}/src/${if ${1},Packages/${1}/}Dataflow/XML
endef

# Return path of TeX directory for package given by ${1}
define texdir
${treetop}/src/${if ${1},Packages/${1}/}Dataflow/TeX
endef

# Make package index.html
define makepkgindex
ruby ${treetop}/doc/Utilities/HTML/makepkgindex.rb ${1} ${2} ${3}
endef 

# ########## Variables

# Top of tree relative to this directory
treetop:=../../..

# Location of stylesheet relative to this directory.
stylesheet:=${treetop}/doc/Utilities/XML/component.xsl

# List of packages
packages:=SCIRun BioPSE Teem VDT MatlabInterface Fusion

# Top of tree relative to SCIRun XML dir
xmlsrtreetop:=../../../

# Top of tree relative to Package XML dirs
xmlpkgtreetop:= ../../../../../

# ########## Top level targets

# Phony targets.
.PHONY: default scirun biopse uintah clean veryclean

# Default target
ifndef SRGROUP
SRGROUP := biopse
endif
ifeq (${strip ${SRGROUP}},)
SRGROUP := biopse
endif
default: ${SRGROUP}

# Other possibilities
scirun : SCIRun

# BioPSE group of packages
biopse : SCIRun BioPSE Teem VDT MatlabInterface Fusion

# Uintah group of packages (This isn't really supported yet).
uintah : SCIRun Unitah

# ########## SCIRun
SCIRun_xml_dir:=${call xmldir}
SCIRun_xml_html_files:=${call xmlhtmlfiles,${SCIRun_xml_dir}}
SCIRun_tex_dir:=${call texdir}
SCIRun_tex_html_dirs:=${call texhtmldirs,${SCIRun_tex_dir}}

SCIRun : SCIRun.html

SCIRun.html : ${SCIRun_xml_html_files} ${SCIRun_tex_html_dirs}
	${call makepkgindex,SCIRun,${SCIRun_xml_dir},${SCIRun_tex_dir}}

${SCIRun_xml_html_files}: ${SCIRun_xml_dir}/%.html: ${SCIRun_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${SCIRun_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

SCIRun_clean:
	-rm -f ${SCIRun_xml_html_files} ${SCIRun_tex_html_files}
	-rm -rf ${SCIRun_tex_html_dirs}

# ########## BioPSE
BioPSE_xml_dir:=${call xmldir,BioPSE}
BioPSE_xml_html_files:=${call xmlhtmlfiles,${BioPSE_xml_dir}}
BioPSE_tex_dir:=${call texdir,BioPSE}
BioPSE_tex_html_dirs:=${call texhtmldirs,${BioPSE_tex_dir}}

BioPSE : BioPSE.html

BioPSE.html : ${BioPSE_xml_html_files} ${BioPSE_tex_html_dirs}
	${call makepkgindex,BioPSE,${BioPSE_xml_dir},${BioPSE_tex_dir}}

${BioPSE_xml_html_files}: ${BioPSE_xml_dir}/%.html: ${BioPSE_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${BioPSE_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

BioPSE_clean:
	-rm -f ${BioPSE_xml_html_files} ${BioPSE_tex_html_files}
	-rm -rf ${BioPSE_tex_html_dirs}

# ########## Teem
Teem_xml_dir:=${call xmldir,Teem}
Teem_xml_html_files:=${call xmlhtmlfiles,${Teem_xml_dir}}
Teem_tex_dir:=${call texdir,Teem}
Teem_tex_html_dirs:=${call texhtmldirs,${Teem_tex_dir}}

Teem : Teem.html

Teem.html : ${Teem_xml_html_files} ${Teem_tex_html_dirs}
	${call makepkgindex,Teem,${Teem_xml_dir},${Teem_tex_dir}}

${Teem_xml_html_files}: ${Teem_xml_dir}/%.html: ${Teem_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${Teem_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Teem_clean:
	-rm -f ${Teem_xml_html_files} ${Teem_tex_html_files}
	-rm -rf ${Teem_tex_html_dirs}

# ########## VDT
VDT_xml_dir:=${call xmldir,VDT}
VDT_xml_html_files:=${call xmlhtmlfiles,${VDT_xml_dir}}
VDT_tex_dir:=${call texdir,VDT}
VDT_tex_html_dirs:=${call texhtmldirs,${VDT_tex_dir}}

VDT : VDT.html

VDT.html : ${VDT_xml_html_files} ${VDT_tex_html_dirs}
	${call makepkgindex,VDT,${VDT_xml_dir},${VDT_tex_dir}}

${VDT_xml_html_files}: ${VDT_xml_dir}/%.html: ${VDT_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${VDT_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

VDT_clean:
	-rm -f ${VDT_xml_html_files} ${VDT_tex_html_files}
	-rm -rf ${VDT_tex_html_dirs}

# ########## MatlabInterface
MatlabInterface_xml_dir:=${call xmldir,MatlabInterface}
MatlabInterface_xml_html_files:=${call xmlhtmlfiles,${MatlabInterface_xml_dir}}
MatlabInterface_tex_dir:=${call texdir,MatlabInterface}
MatlabInterface_tex_html_dirs:=${call texhtmldirs,${MatlabInterface_tex_dir}}

MatlabInterface : MatlabInterface.html

MatlabInterface.html : ${MatlabInterface_xml_html_files} ${MatlabInterface_tex_html_dirs}
	${call makepkgindex,MatlabInterface,${MatlabInterface_xml_dir},${MatlabInterface_tex_dir}}

${MatlabInterface_xml_html_files}: ${MatlabInterface_xml_dir}/%.html: ${MatlabInterface_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${MatlabInterface_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

MatlabInterface_clean:
	-rm -f ${MatlabInterface_xml_html_files} ${MatlabInterface_tex_html_files}
	-rm -rf ${MatlabInterface_tex_html_dirs}

# ########## Fusion
Fusion_xml_dir:=${call xmldir,Fusion}
Fusion_xml_html_files:=${call xmlhtmlfiles,${Fusion_xml_dir}}
Fusion_tex_dir:=${call texdir,Fusion}
Fusion_tex_html_dirs:=${call texhtmldirs,${Fusion_tex_dir}}

Fusion : Fusion.html

Fusion.html : ${Fusion_xml_html_files} ${Fusion_tex_html_dirs}
	${call makepkgindex,Fusion,${Fusion_xml_dir},${Fusion_tex_dir}}

${Fusion_xml_html_files}: ${Fusion_xml_dir}/%.html: ${Fusion_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

${Fusion_tex_html_dirs}: %: %.tex
	${call textohtml,${dir $@}}

Fusion_clean:
	-rm -f ${Fusion_xml_html_files} ${Fusion_tex_html_files}
	-rm -rf ${Fusion_tex_html_dirs}

# ########## Cleanage
clean:
	@echo clean

veryclean: clean ${addsuffix _clean,${packages}}
	-rm ${addsuffix .html,${packages}}

