# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Parallel

SRCS     += \
	$(SRCDIR)/Parallel.cc                \
	$(SRCDIR)/ProcessorGroup.cc          \
	$(SRCDIR)/UintahParallelComponent.cc \
	$(SRCDIR)/UintahParallelPort.cc      \
	$(SRCDIR)/Vampir.cc

PSELIBS := \
	Core/Thread \
	Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

