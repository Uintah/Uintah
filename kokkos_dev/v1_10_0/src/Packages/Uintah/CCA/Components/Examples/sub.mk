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

PSELIBS := \
	Core/Exceptions                   \
	Packages/Uintah/CCA/Ports         \
	Packages/Uintah/Core/Grid         \
	Packages/Uintah/Core/ProblemSpec  \
	Packages/Uintah/Core/Exceptions   \
	Packages/Uintah/Core/Disclosure   \
	Packages/Uintah/Core/Parallel

LIBS := $(XML_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

