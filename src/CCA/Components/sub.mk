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
# 
# 
# 
# Makefile fragment for this subdirectory 


# include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := CCA/Components

# The following variables are used by the Fake* scripts... please
# do not modify...
#


###################################
# We need to add Wasatch to the list of sub-dirs if
# a) building the Watach component, or
# b) building the arches component.
# We have to have the following two separate 'ifeq's because gmake does not support and/or/not... sigh.
#
ifeq ($(BUILD_WASATCH)$(BUILD_WASATCH_FOR_ARCHES),yesno)
  WASATCH := $(SRCDIR)/Wasatch
endif
ifeq ($(BUILD_WASATCH)$(BUILD_WASATCH_FOR_ARCHES),noyes)
  WASATCH := $(SRCDIR)/Wasatch
endif
#
###################################

ifeq ($(BUILD_MPM),yes)
  MPM      := $(SRCDIR)/MPM
  ifeq ($(BUILD_ICE),yes)
    MPMICE := $(SRCDIR)/MPMICE
  endif
endif

ifeq ($(BUILD_ICE),yes)
  ICE      := $(SRCDIR)/ICE
endif

ifeq ($(BUILD_ARCHES),yes)
  ARCHES   := $(SRCDIR)/Arches
	ifeq ($(BUILD_MPM),yes)
		MPMARCHES := $(SRCDIR)/MPMArches
	endif
endif

ifeq ($(BUILD_MD),yes)
  MD      := $(SRCDIR)/MD
endif

SUBDIRS := \
        $(MPM)                         \
        $(ICE)                         \
        $(MPMICE)                      \
        $(ARCHES)                      \
        $(MPMARCHES)                   \
        $(WASATCH)                     \
        $(SRCDIR)/DataArchiver         \
        $(SRCDIR)/Examples             \
        $(SRCDIR)/LoadBalancers        \
        $(SRCDIR)/Models               \
        $(SRCDIR)/OnTheFlyAnalysis     \
        $(SRCDIR)/Parent               \
        $(MD)                          \
        $(SRCDIR)/ProblemSpecification \
        $(SRCDIR)/ReduceUda            \
        $(SRCDIR)/Regridder            \
        $(SRCDIR)/Schedulers           \
        $(SRCDIR)/SimulationController \
        $(SRCDIR)/Solvers              \
        $(SRCDIR)/SwitchingCriteria    


include $(SCIRUN_SCRIPTS)/recurse.mk

