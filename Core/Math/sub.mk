# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Math

SRCS     += \
	$(SRCDIR)/FastMatrix.cc \
	$(SRCDIR)/Primes.cc     \
	$(SRCDIR)/Matrix3.cc    \
	$(SRCDIR)/SymmMatrix3.cc    \
	$(SRCDIR)/CubeRoot.cc	\
	$(SRCDIR)/Sparse.cc	\
	$(SRCDIR)/Short27.cc \
	$(SRCDIR)/TangentModulusTensor.cc \
	$(SRCDIR)/LinearInterpolator.cc \
	$(SRCDIR)/Node27Interpolator.cc 

ifeq ($(IS_WIN),yes)
  SRCS += $(SRCDIR)/Rand48.cc
endif


PSELIBS := \
	Core/Exceptions                 \
	Core/Util                       \
	Core/Geometry                   \
	Core/Thread			


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) 


