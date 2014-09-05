# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

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

LIBS := $(MPI_LIBRARY) $(VAMPIR_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

