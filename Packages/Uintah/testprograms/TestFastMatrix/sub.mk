# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/TestFastMatrix

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)  $(F_LIBRARY) $(BLAS_LIBRARY)

##############################################3
# test fast matrix

SRCS := $(SRCDIR)/testfastmatrix.cc

PSELIBS := Packages/Uintah/Core/Math
PROGRAM := $(SRCDIR)/testfastmatrix

include $(SCIRUN_SCRIPTS)/program.mk

##############################################3
# perf fast matrix

SRCS := $(SRCDIR)/perffastmatrix.cc

PSELIBS := Packages/Uintah/Core/Math Core/Thread
PROGRAM := $(SRCDIR)/perffastmatrix

include $(SCIRUN_SCRIPTS)/program.mk
