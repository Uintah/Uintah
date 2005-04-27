#Makefile fragment for the Packages/Uintah/Core directory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
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

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Core/Geom \
        Core/Datatypes \
        Core/Util



include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
