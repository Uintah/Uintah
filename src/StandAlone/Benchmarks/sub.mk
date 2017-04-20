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
# Makefile fragment for this subdirectory 

SRCDIR := StandAlone/Benchmarks

##############################################
# No subdirectories yet, maybe in the future
#
#SUBDIRS := 
#
#include $(SCIRUN_SCRIPTS)/recurse.mk
#


##############################################
# Simple Math Benchmark

SRCS    := $(SRCDIR)/SimpleMath.cc

PROGRAM := $(SRCDIR)/SimpleMath

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(ALL_STATIC_PSE_LIBS)
else # Non-static build
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := $(ALL_PSE_LIBS)
  endif
endif

PSELIBS := $(GPU_EXTRA_LINK) $(PSELIBS)

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS) $(ZOLTAN_LIBRARY)    \
          $(BOOST_LIBRARY)         \
          $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY) \
          $(TABPROPS_LIBRARY) $(RADPROPS_LIBRARY)  \
          $(PAPI_LIBRARY) $(M_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) \
          $(BLAS_LIBRARY) $(THREAD_LIBRARY) $(CUDA_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

SimpleMath: prereqs StandAlone/Benchmarks/SimpleMath
