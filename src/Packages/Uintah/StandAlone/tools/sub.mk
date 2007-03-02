# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools

SUBDIRS := \
	$(SRCDIR)/compare_mms \
	$(SRCDIR)/dumpfields  \
	$(SRCDIR)/extractors  \
	$(SRCDIR)/pfs         \
	$(SRCDIR)/puda

#	$(SRCDIR)/graphview

include $(SCIRUN_SCRIPTS)/recurse.mk

