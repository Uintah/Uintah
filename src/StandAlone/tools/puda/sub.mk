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

SRCDIR  := StandAlone/tools/puda
PROGRAM := StandAlone/tools/puda/puda

SRCS := \
	$(SRCDIR)/asci.cc        \
	$(SRCDIR)/jacquie.cc     \
	$(SRCDIR)/monica1.cc     \
	$(SRCDIR)/monica2.cc     \
	$(SRCDIR)/jim1.cc        \
	$(SRCDIR)/jim2.cc        \
	$(SRCDIR)/jim3.cc        \
	$(SRCDIR)/PIC.cc         \
	$(SRCDIR)/POL.cc         \
	$(SRCDIR)/AA_MMS.cc      \
	$(SRCDIR)/util.cc        \
	$(SRCDIR)/varsummary.cc  \
	$(SRCDIR)/puda.cc        \
	$(SRCDIR)/GV_MMS.cc      \
	$(SRCDIR)/ER_MMS.cc

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(CORE_STATIC_PSELIBS)
else # Non-static build
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := \
        CCA/Components/ProblemSpecification \
        CCA/Ports          \
        Core/DataArchive   \
        Core/Disclosure    \
        Core/Exceptions    \
        Core/Grid          \
        Core/Math          \
        Core/Parallel      \
        Core/ProblemSpec   \
        Core/Util          \
        Core/Containers    \
        Core/Exceptions    \
        Core/Geometry      \
        Core/OS            \
        Core/Persistent    \
        Core/Thread        \
        Core/Util        
  endif
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY) \
	  $(F_LIBRARY) $(BLAS_LIBRARY) $(THREAD_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

