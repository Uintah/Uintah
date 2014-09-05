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



# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Insight/Dataflow/GUI

PATH_TO_SCIRUN := $(shell cd $(SRCTOP) ; pwd)
PATH_TO_PACKAGE := $(PATH_TO_SCIRUN)/Packages/Insight

CODEGEN := -classpath $(PATH_TO_SCIRUN)/tools/CodeGenerator/java:$(XALAN_PATH) SCIRun.GenerateSCIRunCode 

XMLS_PATH = $(PATH_TO_SCIRUN)/Packages/Insight/Dataflow/Modules/Filters/XML
XMLS :=  $(shell ls $(XMLS_PATH)/sci_*.xml)

INSIGHT_TCL_GEN := $(patsubst $(XMLS_PATH)/sci_%.xml, $(SRCDIR)/%.tcl, $(XMLS))

Packages/Insight/Dataflow/GUI/%.tcl:  Packages/Insight/Dataflow/Modules/Filters/XML/sci_%.xml
	java $(CODEGEN) $(PATH_TO_PACKAGE) $(PATH_TO_PACKAGE)/Dataflow/Modules/Filters/XML/sci_$*.xml $(PATH_TO_PACKAGE)/Core/CodeGenerator/XSL/SCIRun_generateTCL.xsl $(PATH_TO_PACKAGE)/Dataflow/GUI/$*.tcl

SRCS := \
	$(INSIGHT_TCL_GEN) \
	$(SRCDIR)/ImageFileWriter.tcl \
	$(SRCDIR)/ImageReaderFloat2D.tcl \
	$(SRCDIR)/ImageReaderFloat3D.tcl \
	$(SRCDIR)/ImageReaderUChar2D.tcl \
	$(SRCDIR)/ImageReaderUChar3D.tcl \
	$(SRCDIR)/ImageReaderUShort2D.tcl \
	$(SRCDIR)/ImageReaderUShort3D.tcl \
	$(SRCDIR)/ImageToField.tcl \
	$(SRCDIR)/ColorImageReaderUChar2D.tcl\
	$(SRCDIR)/ColorImageReaderFloat2D.tcl\
	$(SRCDIR)/ColorImageReaderUChar3D.tcl\
	$(SRCDIR)/ColorImageReaderFloat3D.tcl\
	$(SRCDIR)/ChooseImage.tcl\
	$(SRCDIR)/ExtractImageFilter.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



