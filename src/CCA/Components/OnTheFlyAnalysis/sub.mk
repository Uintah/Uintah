#
#  The MIT License
#
#  Copyright (c) 1997-2018 The University of Utah
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


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/OnTheFlyAnalysis

# Generic analysis modules only -
# Applicaion specific analysis modules are addd below
SRCS += \
        $(SRCDIR)/AnalysisModuleFactory.cc \
        $(SRCDIR)/AnalysisModule.cc        \
        $(SRCDIR)/lineExtract.cc           \
        $(SRCDIR)/MinMax.cc                \
        $(SRCDIR)/momentumAnalysis.cc      \
        $(SRCDIR)/planeAverage.cc          \
        $(SRCDIR)/planeExtract.cc          \
        $(SRCDIR)/statistics.cc            \
        $(SRCDIR)/FileInfoVar.cc

PSELIBS := \
	CCA/Ports               \
        Core/Disclosure         \
        Core/Exceptions         \
        Core/Geometry           \
        Core/GeometryPiece      \
        Core/Grid               \
        Core/Math               \
        Core/OS                 \
        Core/Parallel           \
        Core/ProblemSpec        \
        Core/Util

# ICE analysis modules
ifeq ($(BUILD_ICE),yes)
  SRCS += \
        $(SRCDIR)/containerExtract.cc \
        $(SRCDIR)/vorticity.cc

  PSELIBS += CCA/Components/ICE/Core \
             CCA/Components/ICE/Materials
endif

# MPM analysis modules
ifeq ($(BUILD_MPM),yes)
  SRCS += \
        $(SRCDIR)/flatPlate_heatFlux.cc \
        $(SRCDIR)/particleExtract.cc

  PSELIBS += CCA/Components/MPM/Core \
             CCA/Components/MPM/Materials
endif

# MPM-ICE analysis modules
ifeq ($(BUILD_MPM)$(BUILD_ICE),yesyes)
  SRCS += \
        $(SRCDIR)/1stLawThermo.cc
endif

# Radiation analysis modules
ifeq ($(BUILD_MODELS_RADIATION),yes)
  SRCS += \
        $(SRCDIR)/radiometer.cc

  PSELIBS += CCA/Components/Models
endif

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

