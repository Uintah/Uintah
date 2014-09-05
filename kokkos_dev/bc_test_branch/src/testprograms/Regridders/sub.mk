#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
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

SRCDIR := testprograms/Regridders

#################################################################
# Benchmark

PROGRAM := $(SRCDIR)/benchmark

SRCS    := $(SRCDIR)/benchmark.cc       \
           $(SRCDIR)/BNRTask.cc         \
           $(SRCDIR)/GBRv2Regridder.cc 

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := CCA/Components/Regridder $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := \
        Core/Exceptions          \
        Core/Geometry            \
        Core/Grid                \
        Core/Math                \
        Core/ProblemSpec	 \
        Core/Util
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(M_LIBRARY) $(MPI_LIBRARY) $(BLAS_LIBRARY) $(THREAD_LIBRARY) \
	  $(XML2_LIBRARY) $(Z_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

#################################################################
# Patch Quality

PROGRAM := $(SRCDIR)/patchquality
SRCS    := $(SRCDIR)/patchquality.cc

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := CCA/Components/Regridder $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := \
        Core/Exceptions          \
        Core/Geometry            \
        Core/Grid                \
        Core/Math                \
        Core/Parallel            \
        Core/Util
endif

include $(SCIRUN_SCRIPTS)/program.mk

#################################################################
# Output Patches

PROGRAM := $(SRCDIR)/outputpatches

SRCS    := $(SRCDIR)/outputpatches.cc   \
           $(SRCDIR)/BNRTask.cc         \
           $(SRCDIR)/GBRv2Regridder.cc 

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := CCA/Components/Regridder $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Grid         \
        Core/Math         \
        Core/Util
endif

include $(SCIRUN_SCRIPTS)/program.mk

