# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/graphview

PROGRAM := $(SRCDIR)/graphview

SRCS := $(SRCDIR)/graphview.cc	$(SRCDIR)/GV_TaskGraph.cc 	\
	$(SRCDIR)/DaVinci.cc

PSELIBS := \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Variables \
	Packages/Uintah/Core/Util \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions \
	Core/Exceptions  \
	Core/Util  \
	Core/Geometry \
	Core/Thread 

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

