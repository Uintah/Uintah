# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Examples

SRCS     += \
	$(SRCDIR)/Poisson1.cc \
	$(SRCDIR)/Poisson2.cc \
	$(SRCDIR)/Burger.cc \
	$(SRCDIR)/Poisson3.cc \
	$(SRCDIR)/SimpleCFD.cc \
	$(SRCDIR)/AMRSimpleCFD.cc \
	$(SRCDIR)/Interpolator.cc \
	$(SRCDIR)/ExamplesLabel.cc \
	$(SRCDIR)/BoundaryConditions.cc \
	$(SRCDIR)/RegionDB.cc

PSELIBS :=  Packages/Uintah/Core/Grid Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports Packages/Uintah/Core/Exceptions \
	Core/Exceptions Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Parallel

LIBS := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

