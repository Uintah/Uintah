# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/RTRTViewer.tcl
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/rtrt/Dataflow/GUI


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

