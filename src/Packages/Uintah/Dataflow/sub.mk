#Makefile fragment for the Packages/Uintah/Dataflow directory

SRCDIR := Packages/Uintah/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules

include $(SCIRUN_SCRIPTS)/recurse.mk
