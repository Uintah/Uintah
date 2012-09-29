#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the \"Software\"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR	:= CCA/Components/MPM

SRCS += $(SRCDIR)/SerialMPM.cc    \
	$(SRCDIR)/RigidMPM.cc     \
	$(SRCDIR)/MPMCommon.cc    \
	$(SRCDIR)/FractureMPM.cc  \
	$(SRCDIR)/ImpMPM.cc       \
	$(SRCDIR)/ShellMPM.cc     \
	$(SRCDIR)/AMRMPM.cc       \
	$(SRCDIR)/SimpleSolver.cc \
	$(SRCDIR)/Solver.cc       \
	$(SRCDIR)/MPMBoundCond.cc \
	$(SRCDIR)/MPMFlags.cc	  \
	$(SRCDIR)/ImpMPMFlags.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/PetscSolver.cc 
else
  SRCS += $(SRCDIR)/FakePetscSolver.cc
endif

SUBDIRS := \
	$(SRCDIR)/ConstitutiveModel \
	$(SRCDIR)/Contact           \
	$(SRCDIR)/ThermalContact    \
	$(SRCDIR)/PhysicalBC        \
	$(SRCDIR)/ParticleCreator   \
	$(SRCDIR)/CohesiveZone   \
	$(SRCDIR)/Crack		    \
	$(SRCDIR)/HeatConduction    \
	$(SRCDIR)/MMS

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	CCA/Components/OnTheFlyAnalysis \
	CCA/Ports           \
	Core/Disclosure     \
	Core/Exceptions     \
	Core/Geometry       \
	Core/GeometryPiece  \
	Core/Grid           \
	Core/Labels         \
	Core/Math           \
	Core/Parallel       \
	Core/ProblemSpec    \
	Core/Thread         \
	Core/Util           

LIBS := $(XML2_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) \
	$(LAPACK_LIBRARY) $(BLAS_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY)

ifeq ($(HAVE_PETSC),yes)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(NO_FORTRAN),yes)
  LIBS := $(LIBS) $(F_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
