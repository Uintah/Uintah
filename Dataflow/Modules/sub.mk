# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Operators    \
	$(SRCDIR)/Selectors    \
	$(SRCDIR)/DataIO       \
	$(SRCDIR)/Visualization \
#[INSERT NEW CATEGORY DIR HERE]

#	$(SRCDIR)/MPMViz        \
#	$(SRCDIR)/Writers       \


include $(SCIRUN_SCRIPTS)/recurse.mk

