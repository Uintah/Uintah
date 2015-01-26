#
#  The MIT License
#
#  Copyright (c) 1997-2015 The University of Utah
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

SRCDIR  := StandAlone/tools/dumpfields
PROGRAM := StandAlone/tools/dumpfields/dumpfields

SRCS    := \
	$(SRCDIR)/dumpfields.cc \
	\
	$(SRCDIR)/utils.cc \
	$(SRCDIR)/Args.cc \
	$(SRCDIR)/FieldSelection.cc \
	\
	$(SRCDIR)/FieldDiags.cc \
	$(SRCDIR)/ScalarDiags.cc \
	$(SRCDIR)/VectorDiags.cc \
	$(SRCDIR)/TensorDiags.cc \
	\
	$(SRCDIR)/FieldDumper.cc \
	$(SRCDIR)/TextDumper.cc \
	$(SRCDIR)/EnsightDumper.cc \
	$(SRCDIR)/InfoDumper.cc \
	$(SRCDIR)/HistogramDumper.cc 

ifeq ($(IS_STATIC_BUILD),yes)

  PSELIBS := $(ALL_STATIC_PSE_LIBS)

else # Non-static build

  ifeq ($(LARGESOS),yes)
    PSELIBS := $(ALL_STATIC_PSE_LIBS)
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
  # (Note, THREAD_LIBRARY must come after BLAS_LIBRARY.)
  LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(F_LIBRARY) \
          $(BLAS_LIBRARY) $(THREAD_LIBRARY) \
	  $(BOOST_LIBRARY) $(CUDA_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

