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

SRCDIR := StandAlone/tools/extractors

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(ALL_STATIC_PSE_LIBS)

else # Non-static build

  ifeq ($(LARGESOS),yes)
    PSELIBS := Packages/Uintah
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
          $(PAPI_LIBRARY) $(M_LIBRARY) $(PIDX_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY)       \
          $(CANTERA_LIBRARY)                                  \
          $(PETSC_LIBRARY)  $(LAPACK_LIBRARY) $(BLAS_LIBRARY) \
          $(MPI_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY)       \
          $(CUDA_LIBRARY) $(PIDX_LIBRARY)
  ifeq ($(HAVE_TIFF),yes)
    LIBS := $(LIBS) $(TIFF_LIBRARY)
  endif
endif

##############################################
# timeextract

SRCS    := $(SRCDIR)/timeextract.cc
PROGRAM := $(SRCDIR)/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# faceextract

SRCS    := $(SRCDIR)/faceextract.cc
PROGRAM := $(SRCDIR)/faceextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractV

SRCS    := $(SRCDIR)/extractV.cc
PROGRAM := $(SRCDIR)/extractV

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractF

SRCS    := $(SRCDIR)/extractF.cc
PROGRAM := $(SRCDIR)/extractF

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractS

SRCS    := $(SRCDIR)/extractS.cc
PROGRAM := $(SRCDIR)/extractS

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# partextract

SRCS    := $(SRCDIR)/partextract.cc
PROGRAM := $(SRCDIR)/partextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract

SRCS    := $(SRCDIR)/lineextract.cc
PROGRAM := $(SRCDIR)/lineextract

include $(SCIRUN_SCRIPTS)/program.mk


##############################################
# particle2tiffls

ifeq ($(HAVE_TIFF),yes)

  SRCS    := $(SRCDIR)/particle2tiff.cc
  PROGRAM := $(SRCDIR)/particle2tiff
  LIBS    := $(LIBS) $(TIFF_LIBRARY)
  include $(SCIRUN_SCRIPTS)/program.mk

endif
