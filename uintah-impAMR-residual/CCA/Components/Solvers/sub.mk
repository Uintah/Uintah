# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Solvers

SRCS     += \
	$(SRCDIR)/CGSolver.cc \
	$(SRCDIR)/DirectSolve.cc \
	$(SRCDIR)/SolverFactory.cc

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/HypreSolver.cc 
  
  SRCS += $(SRCDIR)/AMRSolver.cc \
	$(SRCDIR)/HypreDriver.cc \
	$(SRCDIR)/HypreDriverStruct.cc \
	$(SRCDIR)/HypreDriverSStruct.cc
  SUBDIRS := \
        $(SRCDIR)/HypreSolvers \
        $(SRCDIR)/HyprePreconds

  include $(SCIRUN_SCRIPTS)/recurse.mk

endif # if $(HAVE_HYPRE)

PSELIBS := \
	Core/Containers                  \
	Core/Exceptions                  \
	Core/Thread                      \
	Core/Util                        \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML_LIBRARY) 

ifeq ($(HAVE_HYPRE),yes)
LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

