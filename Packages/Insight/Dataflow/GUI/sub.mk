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

SRCDIR := Packages/Insight/Dataflow/GUI

PATH_TO_SCIRUN := $(shell cd $(SRCTOP) ; pwd)
PATH_TO_PACKAGE := $(PATH_TO_SCIRUN)/Packages/Insight

CODEGEN := -classpath $(PATH_TO_SCIRUN)/tools/CodeGenerator/java:$(XALAN_PATH) SCIRun.GenerateSCIRunCode 

XMLS :=  \
	sci_DiscreteGaussianImageFilter.xml \
        sci_GradientAnisotropicDiffusionImageFilter.xml \
        sci_GradientMagnitudeImageFilter.xml \
	sci_WatershedRelabeler.xml \
	sci_WatershedSegmenter.xml \
	sci_ThresholdSegmentationLevelSetImageFilter.xml \
	sci_BinaryThresholdImageFilter.xml \
	sci_ReflectImageFilter.xml \
	sci_CannySegmentationLevelSetImageFilter.xml \
	sci_WatershedSegmentTreeGenerator.xml \
#[INSERT NEW CODE FILE HERE]

INSIGHT_TCL_GEN := $(patsubst sci_%.xml, $(SRCDIR)/%.tcl, $(XMLS))

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
	$(SRCDIR)/Switch.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



