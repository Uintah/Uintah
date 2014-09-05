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
# Makefile fragment for this subdirectory

SRCDIR := Uintah/StandAlone/tools/compare_mms


ifeq ($(LARGESOS),yes)
  PSELIBS := Uintah
else

  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
        Core/Util         \
        Uintah/Core/DataArchive \
        Uintah/Core/Disclosure  \
        Uintah/Core/Exceptions  \
        Uintah/Core/Grid        \
        Uintah/Core/Labels      \
        Uintah/Core/ProblemSpec                    \
        Uintah/Core/Util                           \
        Uintah/CCA/Components/DataArchiver         \
        Uintah/CCA/Components/Schedulers           \
        Uintah/CCA/Components/ProblemSpecification \
        Uintah/CCA/Components/PatchCombiner
endif

LIBS := $(BLAS_LIBRARY) $(LAPACK_LIBRARY) $(THREAD_LIBRARY)

########################################################
# compare_mms

SRCS := $(SRCDIR)/compare_mms.cc \
        $(SRCDIR)/ExpMMS.cc \
        $(SRCDIR)/LinearMMS.cc \
        $(SRCDIR)/SineMMS.cc 

PROGRAM := $(SRCDIR)/compare_mms


include $(SCIRUN_SCRIPTS)/program.mk


########################################################
# compare_scalar

SRCS := $(SRCDIR)/compare_scalar.cc 
PROGRAM := $(SRCDIR)/compare_scalar



include $(SCIRUN_SCRIPTS)/program.mk


compare_mms: prereqs Uintah/StandAlone/tools/compare_mms/compare_mms
compare_scalar: prereqs Uintah/StandAlone/tools/compare_mms/compare_scalar
