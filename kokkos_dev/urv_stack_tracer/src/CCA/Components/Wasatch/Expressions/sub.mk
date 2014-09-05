#
#  The MIT License
#
#  Copyright (c) 2010-2012 The University of Utah
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

SRCDIR := CCA/Components/Wasatch/Expressions

#
# These are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# WARNING: If you add a file to the list of CUDA_SRCS, you must add a
# corresponding rule at the end of this file!
#
CUDA_ENABLED_SRCS = \
     BasicExprBuilder     \
     ConvectiveFlux       \
     DiffusiveFlux        \
     DiffusiveVelocity    \
     Dilatation           \
     MomentumPartialRHS   \
     MomentumRHS          \
     MonolithicRHS        \
     PoissonExpression    \
     Pressure             \
     PressureSource       \
     PrimVar              \
     ScalarRHS            \
     SolnVarEst           \
     Strain               \
     WeakConvectiveTerm   \
     VelEst               \
     Vorticity                  

ifeq ($(HAVE_CUDA),yes)

   # CUDA enabled files, listed here (and with a rule at the end of
   # this sub.mk) are copied to the binary side and renamed with a .cu
   # extension (.cc replaced with .cu) so that they can be compiled
   # using the nvcc compiler.

   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(OBJTOP_ABS)/$(SRCDIR)/$(var).cu)

else

   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).cc)

endif

#
# Non-CUDA Dependent src files... can be specified the old fashioned
# way:
#
SRCS += \
        $(SRCDIR)/ScalabilityTestSrc.cc   \
        $(SRCDIR)/SetCurrentTime.cc       \
        $(SRCDIR)/RadPropsEvaluator.cc	  \
        $(SRCDIR)/VelocityMagnitude.cc    \
        $(SRCDIR)/StableTimestep.cc       \
        $(SRCDIR)/TabPropsHeatLossEvaluator.cc
            	

#
# Subdirectories to build...
#

SUBDIRS := \
        $(SRCDIR)/MMS \
        $(SRCDIR)/PBE \
        $(SRCDIR)/PostProcessing \
        $(SRCDIR)/Turbulence  \
        $(SRCDIR)/EmbeddedGeometry

include $(SCIRUN_SCRIPTS)/recurse.mk

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # If Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/BasicExprBuilder.cu : $(SRCTOP_ABS)/$(SRCDIR)/BasicExprBuilder.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/ConvectiveFlux.cu : $(SRCTOP_ABS)/$(SRCDIR)/ConvectiveFlux.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/DiffusiveFlux.cu : $(SRCTOP_ABS)/$(SRCDIR)/DiffusiveFlux.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/DiffusiveVelocity.cu : $(SRCTOP_ABS)/$(SRCDIR)/DiffusiveVelocity.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/Dilatation.cu : $(SRCTOP_ABS)/$(SRCDIR)/Dilatation.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/MomentumPartialRHS.cu : $(SRCTOP_ABS)/$(SRCDIR)/MomentumPartialRHS.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/MomentumRHS.cu : $(SRCTOP_ABS)/$(SRCDIR)/MomentumRHS.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/MonolithicRHS.cu : $(SRCTOP_ABS)/$(SRCDIR)/MonolithicRHS.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/PoissonExpression.cu : $(SRCTOP_ABS)/$(SRCDIR)/PoissonExpression.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/Pressure.cu : $(SRCTOP_ABS)/$(SRCDIR)/Pressure.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/PressureSource.cu : $(SRCTOP_ABS)/$(SRCDIR)/PressureSource.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/PrimVar.cu : $(SRCTOP_ABS)/$(SRCDIR)/PrimVar.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/ScalarRHS.cu : $(SRCTOP_ABS)/$(SRCDIR)/ScalarRHS.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/Strain.cu : $(SRCTOP_ABS)/$(SRCDIR)/Strain.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/WeakConvectiveTerm.cu : $(SRCTOP_ABS)/$(SRCDIR)/WeakConvectiveTerm.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/VelEst.cu : $(SRCTOP_ABS)/$(SRCDIR)/VelEst.cc
	cp $< $@

  $(OBJTOP_ABS)/$(SRCDIR)/Vorticity.cu : $(SRCTOP_ABS)/$(SRCDIR)/Vorticity.cc
	cp $< $@

endif
