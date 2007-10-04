# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Solvers

SRCS     += \
	$(SRCDIR)/CGSolver.cc \
	$(SRCDIR)/SolverFactory.cc

PSELIBS := \
	Core/Containers                  \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Thread                      \
	Core/Util                        \
	Core/Geometry                    \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

