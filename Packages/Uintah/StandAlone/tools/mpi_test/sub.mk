# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/mpi_test

##############################################
# async_mpi_test.cc

SRCS    := $(SRCDIR)/async_mpi_test.cc
PROGRAM := $(SRCDIR)/async_mpi_test
PSELIBS := \
        Core/Thread \
        Packages/Uintah/Core/Parallel

LIBS    := $(XML2_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# mpi_test

PROGRAM := $(SRCDIR)/mpi_test
SRCS    := $(SRCDIR)/mpi_test.cc

LIBS    := $(MPI_LIBRARY) $(M_LIBRARY) 
PSELIBS := \
	Core/Containers \
	Core/Exceptions \
	Core/Util       \
	Core/Thread

include $(SCIRUN_SCRIPTS)/program.mk

