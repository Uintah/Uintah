# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools

SUBDIRS := \
	$(SRCDIR)/compare_mms \
	$(SRCDIR)/dumpfields  \
	$(SRCDIR)/extractors  \
        $(SRCDIR)/graphview   \
        $(SRCDIR)/mpi_test    \
	$(SRCDIR)/pfs         \
	$(SRCDIR)/puda        \
	$(SRCDIR)/radiusMaker \
        $(SRCDIR)/uda2nrrd    \
        $(SRCDIR)/uda2vis

include $(SCIRUN_SCRIPTS)/recurse.mk
