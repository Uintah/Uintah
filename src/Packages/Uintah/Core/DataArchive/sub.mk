# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/DataArchive

SRCS += $(SRCDIR)/DataArchive.cc 

PSELIBS := \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)


