# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Math

SRCS     += $(SRCDIR)/Primes.cc	$(SRCDIR)/CubeRoot.cc

PSELIBS := Core/Exceptions
LIBS := -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

