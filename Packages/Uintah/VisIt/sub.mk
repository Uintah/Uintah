
# Makefile fragment for the Packages/Uintah/VisIt directory

SRCDIR := Packages/Uintah/VisIt

SUBDIRS := \
	$(SRCDIR)/udaReaderMTMD

include $(SCIRUN_SCRIPTS)/recurse.mk
