# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Disclosure

SRCS     += \
	$(SRCDIR)/TypeDescription.cc \
	$(SRCDIR)/TypeUtils.cc

PSELIBS := \
	Core/Malloc     \
	Core/Thread     \
	Core/Exceptions \
	Core/Util       \
	Core/Geometry

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

