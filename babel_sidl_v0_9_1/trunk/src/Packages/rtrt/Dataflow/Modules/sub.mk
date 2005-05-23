# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Render    \
	$(SRCDIR)/Scenes    \
#[INSERT NEW CATEGORY DIR HERE]


include $(SCIRUN_SCRIPTS)/recurse.mk

