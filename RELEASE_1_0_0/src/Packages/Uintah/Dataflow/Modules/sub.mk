# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Readers       \
	$(SRCDIR)/Visualization \
#[INSERT NEW CATEGORY DIR HERE]

#	$(SRCDIR)/MPMViz        \
#	$(SRCDIR)/Writers       \


include $(SRCTOP)/scripts/recurse.mk

