# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Readers\
	$(SRCDIR)/Visualization\
#[INSERT NEW CATEGORY DIR HERE]


include $(SRCTOP)/scripts/recurse.mk

