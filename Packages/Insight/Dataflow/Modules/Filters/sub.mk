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

INCLUDES += $(INSIGHT_INCLUDE)

SRCDIR   := Packages/Insight/Dataflow/Modules/Filters

#XMLS :=  $(wildcard $(SRCDIR)/XML/sci_*.xml)
#XMLS :=  $(shell ls $(PATH_TO_SCIRUN)/$(SRCDIR)/XML/sci_*.xml)
XMLS :=  \
	sci_DiscreteGaussianImageFilter.xml \
        sci_GradientAnisotropicDiffusionImageFilter.xml \
        sci_GradientMagnitudeImageFilter.xml \
	sci_WatershedRelabeler.xml \
	sci_WatershedSegmentTreeGenerator.xml \
	sci_WatershedSegmenter.xml \
	sci_CannySegmentationLevelSetImageFilter.xml \
	sci_ThresholdSegmentationLevelSetImageFilter.xml \
	sci_ReflectImageFilter.xml \
	sci_BinaryThresholdImageFilter.xml \
#[INSERT NEW CODE FILE HERE]


SRC_GEN := $(patsubst sci_%.xml, $(SRCDIR)/%.cc, $(XMLS))

CATEGORY := Filters
CODEGEN := Packages/Insight/Core/CodeGenerator/generate

SRCS += ${SRC_GEN} 

$(SRCDIR)/%.cc : $(SRCDIR)/XML/sci_%.xml
	$(CODEGEN) $(PATH_TO_SCIRUN)/Packages/Insight $(CATEGORY) $*
	cp ../src/$@ $@

PSELIBS := Packages/Insight/Core/Datatypes \
	Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions Core/GeomInterface

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


