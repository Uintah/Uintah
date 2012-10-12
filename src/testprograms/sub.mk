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

SRCDIR := testprograms

SUBDIRS := \
        $(SRCDIR)/Malloc                  \
        $(SRCDIR)/Thread                  \
        $(SRCDIR)/TestSuite               \
        $(SRCDIR)/TestFastMatrix          \
        $(SRCDIR)/TestMatrix3             \
        $(SRCDIR)/TestConsecutiveRangeSet \
        $(SRCDIR)/TestRangeTree           \
        $(SRCDIR)/TestBoxGrouper          \
        $(SRCDIR)/BNRRegridder            \
        $(SRCDIR)/Regridders              \
        $(SRCDIR)/IteratorTest            \
        $(SRCDIR)/RegionTest              \
        $(SRCDIR)/CubeRootTest            \
        $(SRCDIR)/PatchBVH

include $(SCIRUN_SCRIPTS)/recurse.mk

PROGRAM := $(SRCDIR)/RunTests
SRCS     = $(SRCDIR)/RunTests.cc

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := \
        testprograms/TestSuite               \
        testprograms/TestMatrix3             \
        testprograms/TestConsecutiveRangeSet \
        testprograms/TestRangeTree           \
        testprograms/TestBoxGrouper          \
        $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := \
        testprograms/TestSuite               \
        testprograms/TestMatrix3             \
        Core/Util                            \
        testprograms/TestConsecutiveRangeSet \
        testprograms/TestRangeTree           \
        testprograms/TestBoxGrouper \
	\
	Core/Math
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) $(THREAD_LIBRARY) $(Z_LIBRARY) \
          $(TEEM_LIBRARY) $(PNG_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk
