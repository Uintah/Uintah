# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Parallel

SRCS     += $(SRCDIR)/Parallel.cc $(SRCDIR)/ProcessorGroup.cc \
	$(SRCDIR)/UintahParallelCore/CCA/Component.cc $(SRCDIR)/UintahParallelPort.cc \
	$(SRCDIR)/Vampir.cc

PSELIBS := Uintah/Grid Core/Thread Core/Exceptions
LIBS := -lmpi $(VAMPIR_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

