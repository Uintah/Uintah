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

CODEGEN := Packages/Insight/Core/CodeGenerator/generate

XMLS :=  \
	sci_DiscreteGaussianImageFilter.xml \
        sci_GradientAnisotropicDiffusionImageFilter.xml \
        sci_GradientMagnitudeImageFilter.xml \
	sci_WatershedRelabeler.xml \
	sci_WatershedSegmentTreeGenerator.xml \
	sci_WatershedSegmenter.xml \
	sci_CannySegmentationLevelSetImageFilter.xml \
	sci_ThresholdSegmentationLevelSetImageFilter.xml \
#[INSERT NEW CODE FILE HERE]

INSIGHT_TCL_GEN := $(patsubst sci_%.xml, $(SRCDIR)/%.tcl, $(XMLS))

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

Packages/Insight/Dataflow/GUI/%.tcl:  Packages/Insight/Dataflow/Modules/Filters/XML/sci_%.xml
	$(CODEGEN) $(PATH_TO_SCIRUN)/Packages/Insight $(CATEGORY) $* -gui 

$(SRCDIR)/tclIndex: $(INSIGHT_TCL_GEN)
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Insight/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


