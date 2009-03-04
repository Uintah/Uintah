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
SRCDIR := Uintah/testprograms

SUBDIRS := \
        $(SRCDIR)/TestSuite               \
        $(SRCDIR)/TestFastMatrix          \
        $(SRCDIR)/TestMatrix3             \
        $(SRCDIR)/TestConsecutiveRangeSet \
        $(SRCDIR)/TestRangeTree           \
        $(SRCDIR)/TestBoxGrouper          \
        $(SRCDIR)/BNRRegridder            \
	$(SRCDIR)/IteratorTest            \
	$(SRCDIR)/PatchBVH

#       $(SRCDIR)/SFCTest \

include $(SCIRUN_SCRIPTS)/recurse.mk

PROGRAM := $(SRCDIR)/RunTests

SRCS    = $(SRCDIR)/RunTests.cc

PSELIBS := \
        Uintah/testprograms/TestSuite               \
        Uintah/testprograms/TestMatrix3             \
        Uintah/Core/Util                            \
        Uintah/testprograms/TestConsecutiveRangeSet \
        Uintah/testprograms/TestRangeTree           \
        Uintah/testprograms/TestBoxGrouper

LIBS := $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) $(THREAD_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk
