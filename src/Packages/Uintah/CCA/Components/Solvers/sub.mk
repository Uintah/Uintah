# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Solvers

SRCS     += \
	$(SRCDIR)/CGSolver.cc \
	$(SRCDIR)/DirectSolve.cc

PSELIBS := \
	Core/Exceptions                  \
	Core/Thread                      \
	Core/Util                        \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML_LIBRARY) 
#$(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

