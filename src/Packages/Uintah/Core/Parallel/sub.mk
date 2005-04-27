# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Parallel

SRCS     += \
	$(SRCDIR)/Parallel.cc                \
	$(SRCDIR)/ProcessorGroup.cc          \
	$(SRCDIR)/UintahParallelComponent.cc \
	$(SRCDIR)/UintahParallelPort.cc      \
	$(SRCDIR)/BufferInfo.cc              \
	$(SRCDIR)/PackBufferInfo.cc          \
	$(SRCDIR)/Vampir.cc

PSELIBS := \
	Core/Thread \
	Core/Exceptions 

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY) 


