# Makefile fragment for this subdirectory

SRCDIR := Packages/Butson/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/DipoleInSphere.tcl

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
