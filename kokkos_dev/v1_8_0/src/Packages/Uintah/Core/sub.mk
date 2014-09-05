#Makefile fragment for the Packages/Uintah/Core directory

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
	$(SRCDIR)/Parallel    \
	$(SRCDIR)/Disclosure  \
	$(SRCDIR)/Math        \
	$(SRCDIR)/Exceptions  \
	$(SRCDIR)/Grid        \
	$(SRCDIR)/DataArchive \
	$(SRCDIR)/ProblemSpec \
	$(SRCDIR)/Datatypes

include $(SCIRUN_SCRIPTS)/recurse.mk

