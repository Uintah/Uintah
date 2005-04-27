# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/makedot

PROGRAM := $(SRCDIR)/makedot

SRCS := $(SRCDIR)/makedot.cc

PSELIBS := \
	Packages/Uintah/Core		\
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Exceptions  \
	Core/Geometry \
	Core/Thread 

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

