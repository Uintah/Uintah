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

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ImageFileReader.tcl\
	$(SRCDIR)/ImageFileWriter.tcl\
	$(SRCDIR)/ImageReaderUChar2D.tcl\
	$(SRCDIR)/Switch.tcl\
	$(SRCDIR)/ImageReaderFloat2D.tcl\
	$(SRCDIR)/ImageReaderFloat3D.tcl\
	$(SRCDIR)/ThresholdSegmentationLevelSetImageFilter.tcl\
#	$(SRCDIR)/DiscreteGaussianImageFilter.tcl\
#	$(SRCDIR)/GradientAnisotropicDiffusionImageFilter.tcl\
#	$(SRCDIR)/CannySegmentationLevelSetImageFilter.tcl\
#	$(SRCDIR)/WatershedSegmenter.tcl\
#	$(SRCDIR)/WatershedSegmentTreeGenerator.tcl\
#	$(SRCDIR)/WatershedRelabeler.tcl\
#	$(SRCDIR)/ImageReaderUShort2D.tcl\
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Insight/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


