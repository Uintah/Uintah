#Makefile fragment for the Packages/Uintah/Core directory

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
	$(SRCDIR)/BoundaryConditions \
	$(SRCDIR)/DataArchive \
	$(SRCDIR)/Datatypes   \
	$(SRCDIR)/Disclosure  \
	$(SRCDIR)/Exceptions  \
	$(SRCDIR)/Grid        \
	$(SRCDIR)/GeometryPiece   \
	$(SRCDIR)/Labels      \
	$(SRCDIR)/Math        \
	$(SRCDIR)/Parallel    \
	$(SRCDIR)/ProblemSpec \
	$(SRCDIR)/Util        \
	$(SRCDIR)/Variables

include $(SCIRUN_SCRIPTS)/recurse.mk

