#
#  The MIT License
#
#  Copyright (c) 2010-2016 The University of Utah
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

SRCDIR   := CCA/Components/Wasatch/Transport

#
# These are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# WARNING: If you add a file to the list of CUDA_ENABLED_SRCS, you must add a
# corresponding rule at the end of this file!
#
CUDA_ENABLED_SRCS =                         \
     CompressibleMomentumTransportEquation 	\
     MomentTransportEquation                \
     MomentumTransportEquationBase          \
     LowMachMomentumTransportEquation       \
     TotalInternalEnergyTransportEquation   \
     ParticleMomentumEquation

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

#
# Non-CUDA Dependent src files... can be specified the old fashioned
# way:
#
SRCS +=                                                \
        $(SRCDIR)/ParseEquation.cc                     \
        $(SRCDIR)/EquationBase.cc                      \
        $(SRCDIR)/ParticleEquationBase.cc              \
        $(SRCDIR)/ParticlePositionEquation.cc          \
        $(SRCDIR)/ParticleMassEquation.cc              \
        $(SRCDIR)/ParticleSizeEquation.cc              \
        $(SRCDIR)/EnthalpyTransportEquation.cc         \
        $(SRCDIR)/ScalabilityTestTransportEquation.cc  \
        $(SRCDIR)/ScalarTransportEquation.cc           \
        $(SRCDIR)/TransportEquation.cc                 \
        $(SRCDIR)/ParseEquationHelper.cc

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # If Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/ParticleMomentumEquation.cu : $(SRCTOP_ABS)/$(SRCDIR)/ParticleMomentumEquation.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/MomentTransportEquation.cu : $(SRCTOP_ABS)/$(SRCDIR)/MomentTransportEquation.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/MomentumTransportEquationBase.cu : $(SRCTOP_ABS)/$(SRCDIR)/MomentumTransportEquationBase.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/LowMachMomentumTransportEquation.cu : $(SRCTOP_ABS)/$(SRCDIR)/LowMachMomentumTransportEquation.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/CompressibleMomentumTransportEquation.cu : $(SRCTOP_ABS)/$(SRCDIR)/CompressibleMomentumTransportEquation.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TotalInternalEnergyTransportEquation.cu : $(SRCTOP_ABS)/$(SRCDIR)/TotalInternalEnergyTransportEquation.cc
	cp $< $@

endif
