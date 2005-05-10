# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Math

SRCS     += \
	$(SRCDIR)/FastMatrix.cc \
	$(SRCDIR)/Primes.cc     \
	$(SRCDIR)/Matrix3.cc    \
	$(SRCDIR)/SymmMatrix3.cc    \
	$(SRCDIR)/CubeRoot.cc	\
	$(SRCDIR)/Sparse.cc	\
	$(SRCDIR)/Short27.cc \
	$(SRCDIR)/TangentModulusTensor.cc 


ifeq ($(IS_WIN),yes)
  SRCS += $(SRCDIR)/Rand48.cc
endif


PSELIBS := \
	Core/Exceptions                 \
	Core/Util                       \
	Core/Geometry                   \
	Core/Thread			\
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Grid


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

