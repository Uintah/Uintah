#Makefile fragment for the Packages/Uintah/Dataflow directory

SRCDIR := Packages/Uintah/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports

include $(SCIRUN_SCRIPTS)/recurse.mk

# This is a target which will only build the libraries that are
# necessary for Uintah SCIRun interaction.  UINTAH_SCIRUN should have
# a list of the libraries that are needed.

uintahmodules: prereqs $(UINTAH_SCIRUN)
