# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Visualization
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

