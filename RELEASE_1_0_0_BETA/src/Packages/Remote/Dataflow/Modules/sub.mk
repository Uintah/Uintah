# Makefile fragment for this subdirectory

SRCDIR := Packages/Remote/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/remoteSalmon\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

