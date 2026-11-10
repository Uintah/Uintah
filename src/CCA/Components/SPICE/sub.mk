#
#  The MIT License
#
#  Copyright (c) 1997-2026 The University of Utah
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
#
# SPICE (Semi-implicit Pressure Iterated Compressible EOS): owns an ICE
# object and schedules its tasks, plus the PITCH / Kurganov-Tadmor tasks.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/SPICE

SRCS   += \
        $(SRCDIR)/SPICE.cc \

PSELIBS := \
	$(SRCDIR)/Advection             \
	$(SRCDIR)/Core                  \
	CCA/Components/Application      \
	CCA/Components/ICE              \
	CCA/Components/ICE/Advection    \
	CCA/Components/ICE/Core         \
	CCA/Components/ICE/CustomBCs    \
	CCA/Components/ICE/EOS          \
	CCA/Components/ICE/Materials    \
	CCA/Components/ICE/TurbulenceModel \
	CCA/Components/Models           \
	CCA/Components/OnTheFlyAnalysis \
	CCA/Ports          \
	Core/Disclosure    \
	Core/Exceptions    \
	Core/Geometry      \
	Core/GeometryPiece \
	Core/Grid          \
	Core/Math          \
	Core/Parallel      \
	Core/ProblemSpec   \
	Core/Util

# the gas-solid interface exchange model needs MPM (FSPICE)
ifeq ($(BUILD_MPM),yes)
  PSELIBS += $(SRCDIR)/Exchange          \
             CCA/Components/MPM/Core      \
             CCA/Components/MPM/Materials
endif

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

ifeq ($(HAVE_HYPRE),yes)
   LIBS := $(LIBS) $(HYPRE_LIBRARY)
   INCLUDES += $(HYPRE_INCLUDE)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


#### Handle subdirs that are their OWN SHARED LIBRARIES
SUBDIRS := \
	$(SRCDIR)/Advection \
	$(SRCDIR)/Core

ifeq ($(BUILD_MPM),yes)
  SUBDIRS += $(SRCDIR)/Exchange
endif

include $(SCIRUN_SCRIPTS)/recurse.mk
#### End handle subdirs
