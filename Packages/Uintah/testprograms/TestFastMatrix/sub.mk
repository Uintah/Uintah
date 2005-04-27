# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/TestFastMatrix

SRCS := $(SRCDIR)/testfastmatrix.cc

PSELIBS := \
	Packages/Uintah/Core \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Math	\
	Core/Geom

PROGRAM := $(SRCDIR)/testfastmatrix

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

SRCS := $(SRCDIR)/perffastmatrix.cc

PSELIBS := \
	Packages/Uintah/Core \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Thread	\
	Core/Math

PROGRAM := $(SRCDIR)/perffastmatrix

LIBS := $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk
