# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/GridVolVis.tcl \
	$(SRCDIR)/GridSliceVis.tcl \
	$(SRCDIR)/SCIRex.tcl \
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Kurt/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

