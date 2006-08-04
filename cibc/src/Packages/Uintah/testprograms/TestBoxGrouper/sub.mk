# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestBoxGrouper

SRCS := $(SRCDIR)/TestBoxGrouper.cc \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/BoxRangeQuerier.cc

PSELIBS := \
	Core/Exceptions \
	Core/Geometry \
	Core/Thread \
	Core/Containers \
	Packages/Uintah/testprograms/TestSuite

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
