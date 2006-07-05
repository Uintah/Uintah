
# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/BNRRegridder

LIBS := $(MPI_LIBRARY) 

PROGRAM := $(SRCDIR)/bnrtest
SRCS := $(SRCDIR)/bnrtest.cc
PSELIBS := Packages/Uintah/CCA/Ports \
           Packages/Uintah/Core/Parallel \
					 Packages/Uintah/CCA/Components/Regridder \
           Core/Thread

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk


