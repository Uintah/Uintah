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


# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

PATH_TO_SCIRUN := $(shell cd $(SRCTOP) ; pwd)
PATH_TO_PACKAGE := $(PATH_TO_SCIRUN)/Packages/Insight

INCLUDES += $(INSIGHT_INCLUDE)

SRCDIR   := Packages/Insight/Dataflow/Modules/Filters

XMLS_PATH := $(PATH_TO_SCIRUN)/Packages/Insight/Dataflow/Modules/Filters/XML
XMLS :=  $(shell ls $(XMLS_PATH)/sci_*.xml)

SRC_GEN := $(patsubst $(XMLS_PATH)/sci_%.xml, $(SRCDIR)/%.cc, $(XMLS))

CATEGORY := Filters
CODEGEN := -classpath $(PATH_TO_SCIRUN)/tools/CodeGenerator/java:$(XALAN_PATH) SCIRun.GenerateSCIRunCode 

SRCS += ${SRC_GEN} \
	${SRCDIR}/ExtractImageFilter.cc \
#[INSERT NEW CODE FILE HERE]

$(SRCDIR)/%.cc : $(SRCDIR)/XML/sci_%.xml
	java $(CODEGEN) $(PATH_TO_PACKAGE) $(PATH_TO_PACKAGE)/Dataflow/Modules/Filters/XML/sci_$*.xml $(PATH_TO_PACKAGE)/Core/CodeGenerator/XSL/SCIRun_generateCC.xsl $(PATH_TO_PACKAGE)/Dataflow/Modules/Filters/$*.cc
	java $(CODEGEN) $(PATH_TO_PACKAGE) $(PATH_TO_PACKAGE)/Dataflow/Modules/Filters/XML/sci_$*.xml $(PATH_TO_PACKAGE)/Core/CodeGenerator/XSL/SCIRun_generateXML.xsl $(PATH_TO_PACKAGE)/Dataflow/XML/$*.xml
	cp $(PATH_TO_SCIRUN)/$@ $@

PSELIBS := Packages/Insight/Core/Datatypes \
	Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions Core/GeomInterface

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


