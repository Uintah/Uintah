# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

SRCDIR := Core/Grid/Variables

SRCS += \
        $(SRCDIR)/Iterator.cc                   \
        $(SRCDIR)/CellIterator.cc               \
        $(SRCDIR)/NodeIterator.cc               \
        $(SRCDIR)/GridIterator.cc               \
        $(SRCDIR)/GridSurfaceIterator.cc        \
        $(SRCDIR)/ListOfCellsIterator.cc        \
        $(SRCDIR)/DifferenceIterator.cc         \
        $(SRCDIR)/UnionIterator.cc              \
        $(SRCDIR)/ComputeSet.cc                 \
        $(SRCDIR)/ComputeSet_special.cc         \
        $(SRCDIR)/GridVariableBase.cc           \
        $(SRCDIR)/LocallyComputedPatchVarMap.cc \
        $(SRCDIR)/ParticleSubset.cc             \
        $(SRCDIR)/ParticleVariableBase.cc       \
        $(SRCDIR)/ParticleVariable_special.cc   \
        $(SRCDIR)/PerPatchBase.cc               \
        $(SRCDIR)/PSPatchMatlGhost.cc           \
        $(SRCDIR)/PSPatchMatlGhostRange.cc      \
        $(SRCDIR)/ReductionVariableBase.cc      \
        $(SRCDIR)/ReductionVariable_special.cc  \
        $(SRCDIR)/ScrubItem.cc                  \
        $(SRCDIR)/SoleVariableBase.cc           \
        $(SRCDIR)/SoleVariable_special.cc       \
        $(SRCDIR)/Stencil7.cc                   \
        $(SRCDIR)/Stencil4.cc                   \
        $(SRCDIR)/Utils.cc                      \
        $(SRCDIR)/ugc_templates.cc              \
        $(SRCDIR)/VarLabel.cc                   \
        $(SRCDIR)/Variable.cc                   

#
#         $(SRCDIR)/ParticleVariable_templates.cc \
#         $(SRCDIR)/NCVariable_templates.cc \
#         $(SRCDIR)/CCVariable_templates.cc \
#         $(SRCDIR)/SFCXVariable_templates.cc \
#         $(SRCDIR)/SFCYVariable_templates.cc \
#         $(SRCDIR)/SFCZVariable_templates.cc
