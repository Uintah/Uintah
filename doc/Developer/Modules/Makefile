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
# Build module html files from their xml equivalents.  Also generate a
# cover page for each package of module descriptions.
#
# Each package has its own directory in which its module description files
# are stored. 
#
# This file uses xalan-java 2.3.1 or higher to process the xml files.
# The environment variable CLASSPATH must point to xalan.jar, xml-api.jar,
#  and xercesImpl.jar.
#

# ########## Macros (Used with ${call ...} function)

#  Convert xml file to html
define xmltohtml
@echo $@
@sed "s#\".*component.dtd#\"${treetop}/doc/Utilities/XML/component.dtd#" $< > ${1}-tmp.xml
@java org.apache.xalan.xslt.Process -IN ${1}-tmp.xml -XSL ${stylesheet} \
-OUT $@ -PARAM treetop ../${treetop} -HTML
@rm ${1}-tmp.xml
endef

# Produce list of html files which correspond to xml files in package
#  given by ${1}
define htmlfiles
${addprefix ${1}/, ${shell cd ${2}; ls -1 *.xml | sed 's,.xml,.html,g'}}
endef

# Make the directory named by ${1}
define makedir
@test -d ${1} || mkdir ${1}
endef

# Return path of package given by ${1}
define xmldir
${treetop}/src/Packages/${1}/Dataflow/XML
endef

# Make package index.html
define makepkgindex
-@cd ${1}; ruby ../${treetop}/doc/Utilities/HTML/makepkgindex.rb
endef 

# ########## Variables

# Top of tree relative to this directory
treetop:=../../../

# Location of stylesheet relative to this directory.
stylesheet:=${treetop}/doc/Utilities/XML/component.xsl

# List of packages
packages:=SCIRun BioPSE Teem VDT MatlabInterface Fusion Uintah

# ########## Top level targets

# Phony targets.
.PHONY: default scirun biopse uintah clean veryclean

# Default
default: biopse

# Other possibilities
scirun : SCIRun

biopse : scirun BioPSE Teem VDT MatlabInterface Fusion

uintah : scirun Unitah

# ########## SCIRun
SCIRun_xml_dir=${treetop}/src/Dataflow/XML
SCIRun_html_files=${addprefix SCIRun/, ${shell cd ${SCIRun_xml_dir}; ls -1 *.xml | sed 's,.xml,.html,g'}}

SCIRun : ${SCIRun_html_files}
	${call makepkgindex,SCIRun}

${SCIRun_html_files}: SCIRun/%.html: ${SCIRun_xml_dir}/%.xml
	${call makedir,SCIRun}
	${call xmltohtml,SCIRun}

# ########## BioPSE
BioPSE_xml_dir=${call xmldir,BioPSE}
BioPSE_html_files=${call htmlfiles,BioPSE,${BioPSE_xml_dir}}
BioPSE : ${BioPSE_html_files}
	${call makepkgindex,BioPSE}

${BioPSE_html_files}: BioPSE/%.html: ${BioPSE_xml_dir}/%.xml
	${call makedir,BioPSE}
	${call xmltohtml,BioPSE}

# ########## Teem
Teem_xml_dir=${call xmldir,Teem}
Teem_html_files=${call htmlfiles,Teem,${Teem_xml_dir}}
Teem : ${Teem_html_files}
	${call makepkgindex,Teem}

${Teem_html_files}: Teem/%.html: ${Teem_xml_dir}/%.xml
	${call makedir,Teem}
	${call xmltohtml,Teem}

# ########## VDT
VDT_xml_dir=${call xmldir,VDT}
VDT_html_files=${call htmlfiles,VDT,${VDT_xml_dir}}
VDT : ${VDT_html_files}
	${call makepkgindex,VDT}

${VDT_html_files}: VDT/%.html: ${VDT_xml_dir}/%.xml
	${call makedir,VDT}
	${call xmltohtml,VDT}

# ########## MatlabInterface
MatlabInterface_xml_dir=${call xmldir,MatlabInterface}
MatlabInterface_html_files=${call htmlfiles,MatlabInterface,${MatlabInterface_xml_dir}}
MatlabInterface : ${MatlabInterface_html_files}
	${call makepkgindex,MatlabInterface}

${MatlabInterface_html_files}: MatlabInterface/%.html: ${MatlabInterface_xml_dir}/%.xml
	${call makedir,MatlabInterface}
	${call xmltohtml,MatlabInterface}

# ########## Fusion
Fusion_xml_dir=${call xmldir,Fusion}
Fusion_html_files=${call htmlfiles,Fusion,${Fusion_xml_dir}}
Fusion : ${Fusion_html_files}
	${call makepkgindex,Fusion}

${Fusion_html_files}: Fusion/%.html: ${Fusion_xml_dir}/%.xml
	${call makedir,Fusion}
	${call xmltohtml,Fusion}

# ########## Cleanage
clean:
	@echo clean

veryclean: clean
	-for p in ${packages};\
	do\
	  rm -rf $$p;\
	done
