# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Labels

SRCS     += \
	$(SRCDIR)/ICELabel.cc \
	$(SRCDIR)/MPMLabel.cc \
	$(SRCDIR)/MPMICELabel.cc

PSELIBS := \
	Core/Exceptions \
	Core/Util \
	Core/Geometry 

LIBS := $(MPI_LIBRARY)


