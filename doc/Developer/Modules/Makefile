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
# Build module html files from their xml equivalents.  Also generate a
# cover page for each package of module descriptions.
#
# Generated html is stored alongside its source xml.
#
# The variable SRGROUP is expected to be set to one of biopse, scirun, or
#  unitah. If its not set or is empty then biopse is assumed.
#

# ########## Macros (Used with ${call ...} function)

#  Convert xml file to html

#
# If using saxon 6.3 or higher to process the xml files then
# the environment variable CLASSPATH must point to saxon.jar.
#
define xmltohtml
@echo $@
@-sed "s#\".*component.dtd#\"${treetop}/doc/Utilities/XML/component.dtd#" $< >${notdir $<}
@-java com.icl.saxon.StyleSheet -w2 ${notdir $<} ${stylesheet} \
treetop=${1} > $@
@-rm ${notdir $<}
endef

# If using xalan-java 2.3.1 or higher to process the xml files then
# the environment variable CLASSPATH must point to xalan.jar, xml-api.jar,
#  and xercesImpl.jar.
#
# define xmltohtml
# @echo $@
# @-sed "s#\".*component.dtd#\"${treetop}/doc/Utilities/XML/component.dtd#" $< >${notdir $<}
# @-java org.apache.xalan.xslt.Process -IN ${notdir $<} -XSL ${stylesheet} \
# -OUT $@ -PARAM treetop ${1} -HTML
# @-rm ${notdir $<}
# endef

# Produce list of html files which correspond to xml files in package
#  given by ${1}
define htmlfiles
${addprefix ${1}/, ${shell cd ${1}; ls -1 *.xml | sed 's,.xml,.html,g'}}
endef

# Return path of package given by ${1}
define xmldir
${treetop}/src/Packages/${1}/Dataflow/XML
endef

# Make package index.html
define makepkgindex
ruby ${treetop}/doc/Utilities/HTML/makepkgindex.rb ${1} ${2}
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
SRGROUP = biopse
endif
ifeq (${strip ${SRGROUP}},)
SRGROUP = biopse
endif
default: ${SRGROUP}

# Other possibilities
scirun : SCIRun

biopse : scirun BioPSE Teem VDT MatlabInterface Fusion

uintah : scirun Unitah

# ########## SCIRun
SCIRun_xml_dir=${treetop}/src/Dataflow/XML
SCIRun_html_files=${addprefix ${SCIRun_xml_dir}/, ${shell cd ${SCIRun_xml_dir}; ls -1 *.xml | sed 's,.xml,.html,g'}}

SCIRun : ${SCIRun_html_files}
	${call makepkgindex,SCIRun,${SCIRun_xml_dir}}

${SCIRun_html_files}: ${SCIRun_xml_dir}/%.html: ${SCIRun_xml_dir}/%.xml
	${call xmltohtml,${xmlsrtreetop}}

SCIRun_clean:
	-rm -f ${SCIRun_html_files}

# ########## BioPSE
BioPSE_xml_dir=${call xmldir,BioPSE}
BioPSE_html_files=${call htmlfiles,${BioPSE_xml_dir}}
BioPSE : ${BioPSE_html_files}
	${call makepkgindex,BioPSE,${BioPSE_xml_dir}}

${BioPSE_html_files}: ${BioPSE_xml_dir}/%.html: ${BioPSE_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

BioPSE_clean:
	-rm -f ${_html_files}

# ########## Teem
Teem_xml_dir=${call xmldir,Teem}
Teem_html_files=${call htmlfiles,${Teem_xml_dir}}
Teem : ${Teem_html_files}
	${call makepkgindex,Teem,${Teem_xml_dir}}

${Teem_html_files}: ${Teem_xml_dir}/%.html: ${Teem_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

Teem_clean:
	-rm -f ${Teem_html_files}

# ########## VDT
VDT_xml_dir=${call xmldir,VDT}
VDT_html_files=${call htmlfiles,${VDT_xml_dir}}
VDT : ${VDT_html_files}
	${call makepkgindex,VDT,${VDT_xml_dir}}

${VDT_html_files}: ${VDT_xml_dir}/%.html: ${VDT_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

VDT_clean:
	-rm -f ${VDT_html_files}

# ########## MatlabInterface
MatlabInterface_xml_dir=${call xmldir,MatlabInterface}
MatlabInterface_html_files=${call htmlfiles,${MatlabInterface_xml_dir}}
MatlabInterface : ${MatlabInterface_html_files}
	${call makepkgindex,MatlabInterface,${MatlabInterface_xml_dir}}

${MatlabInterface_html_files}: ${MatlabInterface_xml_dir}/%.html: ${MatlabInterface_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

MatlabInterface_clean:
	-rm -f ${MatlabInterface_html_files}

# ########## Fusion
Fusion_xml_dir=${call xmldir,Fusion}
Fusion_html_files=${call htmlfiles,${Fusion_xml_dir}}
Fusion : ${Fusion_html_files}
	${call makepkgindex,Fusion,${Fusion_xml_dir}}

${Fusion_html_files}: ${Fusion_xml_dir}/%.html: ${Fusion_xml_dir}/%.xml
	${call xmltohtml,${xmlpkgtreetop}}

Fusion_clean:
	-rm -f ${Fusion_html_files}

# ########## Cleanage
clean:
	@echo clean

veryclean: clean ${addsuffix _clean,${packages}}
	-rm ${addsuffix .html,${packages}}

