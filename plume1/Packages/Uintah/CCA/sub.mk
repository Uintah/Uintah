#Makefile fragment for the Packages/Uintah/Dataflow directory

SRCDIR := Packages/Uintah/CCA
SUBDIRS := \
	$(SRCDIR)/Components \
	$(SRCDIR)/Ports \

include $(SCIRUN_SCRIPTS)/recurse.mk

