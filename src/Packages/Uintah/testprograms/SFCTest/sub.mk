
# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/SFCTest

LIBS := $(MPI_LIBRARY) 

PROGRAM := $(SRCDIR)/sfctest
SRCS := $(SRCDIR)/sfctest.cc
PSELIBS := Packages/Uintah/CCA/Ports \
           Packages/Uintah/Core/Parallel \
           Core/Thread

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk


