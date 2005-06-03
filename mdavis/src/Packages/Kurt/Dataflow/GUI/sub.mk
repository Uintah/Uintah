# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/GUI

SRCS := \
	$(SRCDIR)/GridVolVis.tcl \
	$(SRCDIR)/GridSliceVis.tcl \
	$(SRCDIR)/SCIRex.tcl \
	$(SRCDIR)/HarvardVis.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

KURT_MODULES := $(KURT_MODULES) $(TCLINDEX)

