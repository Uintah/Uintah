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
LIBS := $(XML_LIBRARY) $(TK_LIBRARY) $(GL_LIBS) $(FLIB) $(MPI_LIBRARY) -lm -lz

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

SUBDIRS := \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/testprograms
include $(SCIRUN_SCRIPTS)/recurse.mk
