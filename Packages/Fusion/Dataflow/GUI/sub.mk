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
	$(SRCDIR)/NrrdFieldConverter.tcl\
	$(SRCDIR)/NIMRODConverter.tcl\
	$(SRCDIR)/VULCANConverter.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



