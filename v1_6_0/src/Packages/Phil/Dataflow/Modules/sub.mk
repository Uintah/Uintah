# Makefile fragment for this subdirectory

SRCDIR := Packages/Phil/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Tbon\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

