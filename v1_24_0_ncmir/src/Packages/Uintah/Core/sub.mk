#Makefile fragment for the Packages/Uintah/Core directory

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
	$(SRCDIR)/DataArchive \
	$(SRCDIR)/Datatypes   \
	$(SRCDIR)/Disclosure  \
	$(SRCDIR)/Exceptions  \
	$(SRCDIR)/Grid        \
	$(SRCDIR)/Labels      \
	$(SRCDIR)/Math        \
	$(SRCDIR)/Parallel    \
	$(SRCDIR)/ProblemSpec

include $(SCIRUN_SCRIPTS)/recurse.mk

