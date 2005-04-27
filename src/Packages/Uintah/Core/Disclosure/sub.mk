# Makefile fragment for this subdirectory


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


