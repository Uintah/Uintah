# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools

SUBDIRS := $(SRCDIR)/dumpfields
#	$(SRCDIR)/graphview

include $(SCIRUN_SCRIPTS)/recurse.mk

