# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Exceptions

SRCS     += $(SRCDIR)/InvalidGrid.cc $(SRCDIR)/InvalidValue.cc \
	$(SRCDIR)/ParameterNotFound.cc \
	$(SRCDIR)/ProblemSetupException.cc \
	$(SRCDIR)/TypeMismatchException.cc  \
	$(SRCDIR)/InvalidCompressionMode.cc

PSELIBS := Core/Exceptions
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

