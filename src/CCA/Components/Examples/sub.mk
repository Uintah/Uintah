#
#  The MIT License
#
#  Copyright (c) 1997-2017 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

SRCDIR := CCA/Components/Examples

SRCS += \
        $(SRCDIR)/AMRHeat.cpp          \
        $(SRCDIR)/AMRWave.cc           \
        $(SRCDIR)/Benchmark.cc         \
        $(SRCDIR)/Burger.cc            \
        $(SRCDIR)/DOSweep.cc           \
        $(SRCDIR)/ExamplesLabel.cc     \
        $(SRCDIR)/Interpolator.cc      \
        $(SRCDIR)/Heat.cpp             \
        $(SRCDIR)/ParticleTest1.cc     \
        $(SRCDIR)/Poisson1.cc          \
        $(SRCDIR)/Poisson2.cc          \
        $(SRCDIR)/Poisson3.cc          \
        $(SRCDIR)/Poisson4.cc          \
        $(SRCDIR)/RegionDB.cc          \
        $(SRCDIR)/RegridderTest.cc     \
        $(SRCDIR)/SolverTest1.cc       \
        $(SRCDIR)/Wave.cc              

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/SolverTest2.cc     
endif

ifeq ($(BUILD_MODELS_RADIATION),yes)
  SRCS += $(SRCDIR)/RMCRT_Test.cc       
endif

ifeq ($(HAVE_CUDA),yes)
  SRCS += $(SRCDIR)/PoissonGPU1.cc                 \
          $(SRCDIR)/PoissonGPU1Kernel.cu           \
          $(SRCDIR)/UnifiedSchedulerTest.cc        \
          $(SRCDIR)/UnifiedSchedulerTestKernel.cu
  DLINK_FILES += \
          CCA/Components/Examples/PoissonGPU1Kernel.o           \
          CCA/Components/Examples/UnifiedSchedulerTestKernel.o 
endif

PSELIBS := \
        CCA/Components/Models \
        CCA/Ports             \
        Core/DataArchive      \
        Core/Disclosure       \
        Core/Exceptions       \
        Core/Geometry         \
        Core/GeometryPiece    \
        Core/Grid             \
        Core/Math             \
        Core/Parallel         \
        Core/ProblemSpec      \
        Core/Util             

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(CUDA_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

