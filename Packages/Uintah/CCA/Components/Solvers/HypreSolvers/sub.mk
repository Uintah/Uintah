# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Solvers/HypreSolvers

SRCS += $(SRCDIR)/HypreSolverBase.cc \
        $(SRCDIR)/HypreSolverPFMG.cc \
        $(SRCDIR)/HypreSolverSMG.cc \
        $(SRCDIR)/HypreSolverSparseMSG.cc \
        $(SRCDIR)/HypreSolverCG.cc \
        $(SRCDIR)/HypreSolverHybrid.cc \
        $(SRCDIR)/HypreSolverGMRES.cc \
        $(SRCDIR)/HypreSolverAMG.cc

ifeq ($(HAVE_HYPRE_1_9),yes)

  SRCS += $(SRCDIR)/HypreSolverFAC.cc

endif # if HAVE_HYPRE_1_9
