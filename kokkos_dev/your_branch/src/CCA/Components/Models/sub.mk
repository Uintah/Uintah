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

SRCDIR   := CCA/Components/Models

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCS    += \
       $(SRCDIR)/ModelFactory.cc

RADIATION :=

ifeq ($(BUILD_MODELS_RADIATION),yes)
  RADIATION += $(SRCDIR)/Radiation
endif


SUBDIRS := $(SRCDIR)/FluidsBased \
           $(RADIATION)

PSELIBS :=                 \
        Core/Exceptions    \
        Core/Geometry      \
        Core/Thread        \
        Core/Util          \
        CCA/Ports          \
        Core/Disclosure    \
        Core/Exceptions    \
        Core/Grid          \
        Core/IO            \
        Core/Util          \
        Core/GeometryPiece \
        Core/Labels        \
        Core/Parallel      \
        Core/ProblemSpec   \
        Core/Math

ifneq ($(BUILD_ICE),no)
  PSELIBS += CCA/Components/ICE
endif

ifneq ($(BUILD_MPM),no)
  PSELIBS += CCA/Components/MPM
endif

ifneq ($(BUILD_MPM),no) 
  ifneq ($(BUILD_ICE),no) 
    PSELIBS += CCA/Components/MPMICE
    SUBDIRS += $(SRCDIR)/HEChem
    SUBDIRS += $(SRCDIR)/SolidReactionModel
  endif
endif

######################################################
# Sub-dir recurse has to be after above to make sure
# we include the right sub-dirs.
#
include $(SCIRUN_SCRIPTS)/recurse.mk
#
######################################################

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY)

ifneq ($(HAVE_CUDA),)
  LIBS :=  $(LIBS) $(CUDA_LIBRARY)
endif

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
