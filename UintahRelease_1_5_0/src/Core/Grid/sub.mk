#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

SRCDIR   := Core/Grid

SUBDIRS := \
        $(SRCDIR)/BoundaryConditions \
        $(SRCDIR)/Variables \
        $(SRCDIR)/PatchBVH

include $(SCIRUN_SCRIPTS)/recurse.mk          

SRCS += \
        $(SRCDIR)/AMR.cc                   \
        $(SRCDIR)/AMR_CoarsenRefine.cc     \
        $(SRCDIR)/Box.cc                   \
        $(SRCDIR)/BSplineInterpolator.cc   \
        $(SRCDIR)/DbgOutput.cc             \
        $(SRCDIR)/Ghost.cc                 \
        $(SRCDIR)/Grid.cc                  \
        $(SRCDIR)/Level.cc                 \
        $(SRCDIR)/LinearInterpolator.cc    \
        $(SRCDIR)/Material.cc              \
        $(SRCDIR)/Node27Interpolator.cc    \
        $(SRCDIR)/PatchRangeTree.cc        \
        $(SRCDIR)/Patch.cc                 \
        $(SRCDIR)/Region.cc                \
        $(SRCDIR)/SimpleMaterial.cc        \
        $(SRCDIR)/SimulationState.cc       \
        $(SRCDIR)/SimulationTime.cc        \
        $(SRCDIR)/Task.cc                  \
        $(SRCDIR)/TOBSplineInterpolator.cc \
        $(SRCDIR)/UnknownVariable.cc       \
        $(SRCDIR)/cpdiInterpolator.cc      \
        $(SRCDIR)/fastCpdiInterpolator.cc  \
        $(SRCDIR)/axiCpdiInterpolator.cc   \
        $(SRCDIR)/fastAxiCpdiInterpolator.cc

PSELIBS := \
        Core/Geometry    \
        Core/Exceptions  \
        Core/Thread      \
        Core/Util        \
        Core/Containers  \
        Core/Parallel    \
        Core/ProblemSpec \
        Core/Exceptions  \
        Core/Math        \
        Core/Disclosure

LIBS := $(MPI_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
