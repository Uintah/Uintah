# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Solvers

SRCS     += \
	$(SRCDIR)/CGSolver.cc

PSELIBS := Packages/Uintah/CCA/Ports

LIBS := -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

