# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Fusion/Dataflow/GUI

SRCS := \
	$(SRCDIR)/FusionSlicePlot.tcl\
	$(SRCDIR)/FusionFieldReader.tcl\
	$(SRCDIR)/FusionFieldSetReader.tcl\
	$(SRCDIR)/NIMRODConverter.tcl\
	$(SRCDIR)/NrrdFieldConverter.tcl\
	$(SRCDIR)/MDSPlusFieldReader.tcl\
	$(SRCDIR)/Plot2DViewer.tcl\
#	$(SRCDIR)/ReactionDiffusion.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



