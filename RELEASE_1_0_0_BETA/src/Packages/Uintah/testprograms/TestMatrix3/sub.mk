# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestMatrix3

SRCS := $(SRCDIR)/testmatrix3.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/testprograms/TestSuite \
	Core/Exceptions \
	Core/Thread \
	Core/Geometry \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

