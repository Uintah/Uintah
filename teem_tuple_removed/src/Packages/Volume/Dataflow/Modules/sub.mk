# Makefile fragment for this subdirectory

SRCDIR := Packages/Volume/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Visualization
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

