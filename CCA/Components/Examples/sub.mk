# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Examples

SRCS     += \
	$(SRCDIR)/Poisson1.cc \
	$(SRCDIR)/Poisson2.cc \
	$(SRCDIR)/Burger.cc \
	$(SRCDIR)/Poisson3.cc \
	$(SRCDIR)/Interpolator.cc \
	$(SRCDIR)/ExamplesLabel.cc

PSELIBS :=  Packages/Uintah/Core/Grid Packages/Uintah/Core/ProblemSpec \
	Core/Exceptions Packages/Uintah/CCA/Ports

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

