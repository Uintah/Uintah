#Makefile fragment for the Packages/rtrt/Dataflow directory

SRCDIR := Packages/rtrt/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules

include $(SCIRUN_SCRIPTS)/recurse.mk
