#Makefile fragment for the Packages/Uintah directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Uintah
SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/Dataflow     \
	$(SRCDIR)/CCA          \
	$(SRCDIR)/tools

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := Core Dataflow
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

SUBDIRS := \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/testprograms
include $(SCIRUN_SCRIPTS)/recurse.mk
