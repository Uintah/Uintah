#Makefile fragment for the Packages/rtrt/Dataflow directory

SRCDIR := Packages/rtrt/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports

include $(SCIRUN_SCRIPTS)/recurse.mk
