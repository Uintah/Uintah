# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/TestFastMatrix

SRCS := $(SRCDIR)/testfastmatrix.cc

PSELIBS := \
	Packages/Uintah/Core/Math
PROGRAM := $(SRCDIR)/testfastmatrix

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

SRCS := $(SRCDIR)/perffastmatrix.cc

PSELIBS := \
	Packages/Uintah/Core/Math Core/Thread
PROGRAM := $(SRCDIR)/perffastmatrix

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk
