#
#  The MIT License
#
#  Copyright (c) 1997-2020 The University of Utah
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

SRCDIR  := StandAlone/tools/puda
PROGRAM := StandAlone/tools/puda/puda

SRCS := \
        $(SRCDIR)/AA_MMS.cc            \
        $(SRCDIR)/asci.cc              \
        $(SRCDIR)/ER_MMS.cc            \
        $(SRCDIR)/gridStats.cc         \
        $(SRCDIR)/GV_MMS.cc            \
        $(SRCDIR)/ICE_momentum.cc      \
        $(SRCDIR)/jacquie.cc           \
        $(SRCDIR)/jim1.cc              \
        $(SRCDIR)/jim2.cc              \
        $(SRCDIR)/DOP.cc               \
        $(SRCDIR)/PIC.cc               \
        $(SRCDIR)/POL.cc               \
        $(SRCDIR)/printCellStresses.cc \
        $(SRCDIR)/printParticleVar.cc  \
        $(SRCDIR)/pressure.cc          \
        $(SRCDIR)/pStressHistogram.cc  \
        $(SRCDIR)/puda.cc              \
        $(SRCDIR)/todd1.cc             \
        $(SRCDIR)/util.cc              \
        $(SRCDIR)/varsummary.cc 
  

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
  LIBS := $(CORE_STATIC_LIBS) $(ZOLTAN_LIBRARY)        \
          $(BOOST_LIBRARY)                             \
          $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY)     \
          $(TABPROPS_LIBRARY) $(RADPROPS_LIBRARY)      \
          $(M_LIBRARY) $(PIDX_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)  \
          $(F_LIBRARY) $(BLAS_LIBRARY)                 \
          $(CUDA_LIBRARY) $(PIDX_LIBRARY)              \
          $(KOKKOS_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

###########################################################
#
# grid_reader
#
# Program used to display a binary grid.xml file in human readable (XML) format.
#
LIBS := 
PSELIBS :=

SRCS    := $(SRCDIR)/grid_reader.cc
PROGRAM := StandAlone/tools/puda/grid_reader

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS) $(ZOLTAN_LIBRARY)        \
          $(BOOST_LIBRARY)                             \
          $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY)     \
          $(TABPROPS_LIBRARY) $(RADPROPS_LIBRARY)      \
          $(M_LIBRARY) $(PIDX_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(KOKKOS_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk


