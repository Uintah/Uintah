# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Exceptions

SRCS     += $(SRCDIR)/InvalidGrid.cc $(SRCDIR)/InvalidValue.cc \
	$(SRCDIR)/ParameterNotFound.cc \
	$(SRCDIR)/ProblemSetupException.cc \
	$(SRCDIR)/TypeMismatchException.cc $(SRCDIR)/UnknownVariable.cc

PSELIBS := Core/Exceptions Uintah/Grid
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

