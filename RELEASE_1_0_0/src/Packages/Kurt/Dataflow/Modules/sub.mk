# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Vis\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

