#Makefile fragment for the Packages/Uintah directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Uintah
SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/Dataflow     \
	$(SRCDIR)/CCA

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := Core Dataflow
LIBS := $(XML_LIBRARY) $(TK_LIBRARY) $(GL_LIBS) $(FLIB) $(MPI_LIBRARY) -lm -lz

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk

SUBDIRS := \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/testprograms
include $(SRCTOP_ABS)/scripts/recurse.mk
