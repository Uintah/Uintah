# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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
# http://software.sci.utah.edu/doc/Developer/Guide/create_module.html
#
# Makefile fragment for this subdirectory
# $Id: sub.mk 40521 2008-03-19 21:06:39Z dav $
#
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/Arches/Mixing/TabProps

SRCS := \
        $(SRCDIR)/BSpline.cc \
        $(SRCDIR)/LU.cc \
        $(SRCDIR)/StateTable.cc

PSELIBS := \
    Core/Exceptions \
    Core/Geometry   \
    Core/Thread

# variables HDF5_LIBRARY, BLAS_LIBRARY, etc are set in configVars.mk
# everything compiles OK with either LIBS var, I don't think the blas/lapack libraries make a difference
# (maybe check with BLAS and LAPACK libraries soon - use w/ DQMOM)

#LIBS := $(HDF5_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY)
LIBS := $(HDF5_LIBRARY)
INCLUDES += $(HDF5_INCLUDE)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
