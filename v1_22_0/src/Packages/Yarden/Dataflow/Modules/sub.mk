# Makefile fragment for this subdirectory

SRCDIR := Packages/Yarden/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Visualization\
#	$(SRCDIR)/Readers\
#	$(SRCDIR)/Writers\
#[INSERT NEW CATEGORY DIR HERE]


include $(SRCTOP)/scripts/recurse.mk

