# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Math

SRCS     += \
	$(SRCDIR)/Primes.cc     \
	$(SRCDIR)/Matrix3.cc    \
	$(SRCDIR)/CubeRoot.cc	\
	$(SRCDIR)/Sparse.cc

PSELIBS := \
	Core/Exceptions                 \
	Core/Util                       \
	Core/Geometry                   \
	Packages/Uintah/Core/Disclosure 


LIBS := -lm $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

