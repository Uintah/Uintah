# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/BioPSE/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ConductivitySearch.tcl\
	$(SRCDIR)/DipoleSearch.tcl\
	$(SRCDIR)/SurfaceToSurface.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Packages/BioPSE/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


