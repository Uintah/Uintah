#Makefile fragment for the Packages/Uintah/Dataflow directory

SRCDIR := Packages/Uintah/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules

include $(SRCTOP_ABS)/scripts/recurse.mk
