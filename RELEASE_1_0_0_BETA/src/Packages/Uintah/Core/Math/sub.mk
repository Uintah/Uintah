# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Math

SRCS     += $(SRCDIR)/Primes.cc	$(SRCDIR)/CubeRoot.cc

PSELIBS := Core/Exceptions
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

