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

SRCDIR := StandAlone/tools

SUBDIRS := \
        $(SRCDIR)/compare_mms \
        $(SRCDIR)/dumpfields  \
        $(SRCDIR)/extractors  \
        $(SRCDIR)/graphview   \
        $(SRCDIR)/mpi_test    \
        $(SRCDIR)/fsspeed     \
        $(SRCDIR)/pfs         \
        $(SRCDIR)/puda        \
        $(SRCDIR)/tracker     

ifeq ($(HAVE_TEEM),yes)
  SUBDIRS += $(SRCDIR)/uda2nrrd \
             $(SRCDIR)/radiusMaker 
endif 

ifeq ($(BUILD_VISIT),yes)
        ifeq ($(HAVE_TEEM),yes)
                SUBDIRS += $(SRCDIR)/uda2vis
        endif 
endif

########################################################
# compute_Lnorm_udas

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else

  PSELIBS := \
        Core/Exceptions   \
        Core/Containers   \
        Core/DataArchive  \
	Core/Grid         \
	Core/Math         \
	Core/Thread       \
	Core/Util         \
        CCA/Components/DataArchiver         
endif

LIBS :=

SRCS := $(SRCDIR)/compute_Lnorm_udas.cc 
PROGRAM := $(SRCDIR)/compute_Lnorm_udas

include $(SCIRUN_SCRIPTS)/program.mk

compute_Lnorm_udas: prereqs StandAlone/tools/compute_Lnorm_udas

include $(SCIRUN_SCRIPTS)/recurse.mk
