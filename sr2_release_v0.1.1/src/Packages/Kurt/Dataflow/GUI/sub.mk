# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/GUI

SRCS := \
	$(SRCDIR)/ParticleFlow.tcl\
#[INSERT NEW TCL FILE HERE]
# 	$(SRCDIR)/GridVolVis.tcl \
# 	$(SRCDIR)/GridSliceVis.tcl \
# 	$(SRCDIR)/SCIRex.tcl \
# 	$(SRCDIR)/HarvardVis.tcl \

include $(SCIRUN_SCRIPTS)/tclIndex.mk

KURT_MODULES := $(KURT_MODULES) $(TCLINDEX)

