# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Nrrd/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/NrrdReader.tcl\
	$(SRCDIR)/NrrdWriter.tcl\
	$(SRCDIR)/SlicePicker.tcl\
#[INSERT NEW TCL FILE HERE]

	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Packages/Nrrd/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
