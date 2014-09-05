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

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/FusionFieldReader.tcl\
	$(SRCDIR)/FusionFieldSetReader.tcl\
	$(SRCDIR)/MDSPlusFieldReader.tcl\
	$(SRCDIR)/MDSPlusDataReader.tcl\
	$(SRCDIR)/Plot2DViewer.tcl\
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Fusion/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


