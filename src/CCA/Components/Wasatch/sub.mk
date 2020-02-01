#
#  The MIT License
#
#  Copyright (c) 2010-2018 The University of Utah
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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/Wasatch

#
# These are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# Do not put the .cc on the file name as the .cc or .cu will be added automatically
# as needed.
#

# Do not put the extension on files in this list as the .cc or .cu will
# be added automatically as needed.
CUDA_ENABLED_SRCS :=    \
      TimeStepper       \
      NestedGraphHelper \
      Wasatch

ifeq ($(HAVE_CUDA),yes)
   # CUDA enabled files, listed here (and with a rule at the end of
   # this sub.mk) are copied to the binary side and renamed with a .cu
   # extension (.cc replaced with .cu) so that they can be compiled
   # using the nvcc compiler.
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(OBJTOP_ABS)/$(SRCDIR)/$(var).cu)
   DLINK_FILES := $(DLINK_FILES) $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).o)
else
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).cc)
endif

SRCS +=                                              \
        $(SRCDIR)/BCHelper.cc                        \
        $(SRCDIR)/ConvectiveInterpolationMethods.cc  \
        $(SRCDIR)/CoordinateHelper.cc                \
        $(SRCDIR)/FieldAdaptor.cc                    \
        $(SRCDIR)/GraphHelperTools.cc                \
        $(SRCDIR)/OldVariable.cc                     \
        $(SRCDIR)/ParticlesHelper.cc                 \
        $(SRCDIR)/ParseTools.cc                      \
        $(SRCDIR)/Properties.cc                      \
        $(SRCDIR)/ReductionHelper.cc                 \
        $(SRCDIR)/TagNames.cc                        \
        $(SRCDIR)/TaskInterface.cc                   \
        $(SRCDIR)/WasatchBCHelper.cc                 \
        $(SRCDIR)/DualTimeMatrixManager.cc           \
        $(SRCDIR)/WasatchParticlesHelper.cc          

PSELIBS :=                        \
        CCA/Components/Application \
        CCA/Components/Models     \
        CCA/Components/Schedulers \
        CCA/Ports                 \
        Core/Disclosure           \
        Core/Exceptions           \
        Core/Geometry             \
        Core/GeometryPiece        \
        Core/Grid                 \
        Core/IO                   \
        Core/Math                 \
        Core/Parallel             \
        Core/ProblemSpec          \
        Core/Util

LIBS := $(Z_LIBRARY) $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)    \
        $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY)                    \
        $(RADPROPS_LIBRARY) $(TABPROPS_LIBRARY)                     \
        $(NSCBC_LIBRARY)                                           \
        $(POKITT_LIBRARY)                                           \
        $(BOOST_LIBRARY) $(LAPACK_LIBRARY) $(BLAS_LIBRARY)

INCLUDES := $(INCLUDES) $(SPATIALOPS_INCLUDE) $(EXPRLIB_INCLUDE)     \
            $(TABPROPS_INCLUDE) $(RADPROPS_INCLUDE) $(NSCBC_INCLUDE) \
            $(POKITT_INCLUDE) $(BOOST_INCLUDE) $(LAPACK_INCLUDE)

########################################################################
#
# Create rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename them with a .cu extension so they can be compiled by the NVCC
# compiler.
#

ifeq ($(HAVE_CUDA),yes)

  # Call the make-cuda-target function on each of the CUDA_ENABLED_SRCS:
  $(foreach file,$(CUDA_ENABLED_SRCS),$(eval $(call make-cuda-target,$(file))))

endif

########################################################################
#
# Sub-dir recurse...

SUBDIRS :=                         \
           $(SRCDIR)/Expressions   \
           $(SRCDIR)/Operators     \
           $(SRCDIR)/Transport     

ifeq ($(HAVE_POKITT),yes)
  SUBDIRS += $(SRCDIR)/Coal
endif

include $(SCIRUN_SCRIPTS)/recurse.mk

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
