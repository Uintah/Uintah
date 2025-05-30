#
#  The MIT License
#
#  Copyright (c) 1997-2025 The University of Utah
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

# Generic models only -
# Applicaion specific models are added below
SUBDIRS :=

ifeq ($(BUILD_MODELS_RADIATION),yes)
  SUBDIRS += $(SRCDIR)/Radiation
endif

PSELIBS :=                 \
        CCA/Ports          \
        CCA/Components/Schedulers \
        Core/Disclosure    \
        Core/Exceptions    \
        Core/Geometry      \
        Core/GeometryPiece \
        Core/Grid          \
        Core/IO            \
        Core/Math          \
        Core/Parallel      \
        Core/ProblemSpec   \
        Core/Util          

# ICE Models
ifeq ($(BUILD_ICE),yes)
  SUBDIRS += $(SRCDIR)/FluidsBased \
             $(SRCDIR)/ParticleBased

  PSELIBS += CCA/Components/ICE/Core      \
	     CCA/Components/ICE/CustomBCs \
	     CCA/Components/ICE/Materials
endif

# MPM Models
ifeq ($(BUILD_MPM),yes)
  PSELIBS += CCA/Components/MPM/Core      \
	     CCA/Components/MPM/Materials
endif

# MPM - ICE Models
ifeq ($(BUILD_MPM)$(BUILD_ICE),yesyes)
  PSELIBS += CCA/Components/MPMICE/Core

  SUBDIRS += $(SRCDIR)/HEChem             \
             $(SRCDIR)/MultiMatlExchange  \
             $(SRCDIR)/SolidReactionModel

endif

######################################################
# Sub-dir recurse has to be after above to make sure
# we include the right sub-dirs.
#
include $(SCIRUN_SCRIPTS)/recurse.mk
#
######################################################

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) \
        $(BLAS_LIBRARY) $(LAPACK_LIBRARY)

ifeq ($(HAVE_CUDA),yes)
  LIBS :=  $(LIBS) $(CUDA_LIBRARY)
endif

ifeq ($(HAVE_PETSC),yes)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifeq ($(HAVE_HYPRE),yes)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
