# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Exceptions

SRCS     += \
	$(SRCDIR)/InvalidGrid.cc            \
	$(SRCDIR)/InvalidValue.cc           \
	$(SRCDIR)/ParameterNotFound.cc      \
	$(SRCDIR)/ProblemSetupException.cc  \
	$(SRCDIR)/TypeMismatchException.cc  \
	$(SRCDIR)/InvalidCompressionMode.cc \
	$(SRCDIR)/VariableNotFoundInGrid.cc

PSELIBS := \
	Core/Exceptions \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

