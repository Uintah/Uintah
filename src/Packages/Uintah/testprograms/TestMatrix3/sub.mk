# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestMatrix3

SRCS := $(SRCDIR)/testmatrix3.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports              \
	Packages/Uintah/Core/Grid              \
	Packages/Uintah/Core/Parallel          \
	Packages/Uintah/Core/Exceptions        \
	Packages/Uintah/Core/Math              \
	Packages/Uintah/testprograms/TestSuite \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

