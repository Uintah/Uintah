#Makefile fragment for the Packages/rtrt/Dataflow directory

SRCDIR := Packages/rtrt/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports

include $(SCIRUN_SCRIPTS)/recurse.mk

# This is a target which will only build the libraries that are
# necessary for rtrt SCIRun interaction.  RTRT_SCIRUN should have
# a list of the libraries that are needed.

rtrtmodules: prereqs $(RTRT_SCIRUN)
