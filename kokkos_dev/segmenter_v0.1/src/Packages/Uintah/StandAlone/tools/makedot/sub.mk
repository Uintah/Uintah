# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/makedot

PROGRAM := $(SRCDIR)/makedot

SRCS := $(SRCDIR)/makedot.cc

PSELIBS := \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions \
	Core/Exceptions  \
	Core/Geometry \
	Core/Thread 

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

