# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools

SUBDIRS := \
	$(SRCDIR)/compare_mms \
	$(SRCDIR)/dumpfields  \
	$(SRCDIR)/extractors  \
        $(SRCDIR)/graphview   \
	$(SRCDIR)/pfs         \
	$(SRCDIR)/puda        \
	$(SRCDIR)/radiusMaker \
        $(SRCDIR)/uda2nrrd

include $(SCIRUN_SCRIPTS)/recurse.mk
