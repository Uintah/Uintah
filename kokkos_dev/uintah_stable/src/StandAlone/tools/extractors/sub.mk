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

SRCDIR := Packages/Uintah/StandAlone/tools/extractors

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else
  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
	Core/Persistent   \
        Core/Thread       \
        Core/Util         \
        Packages/Uintah/Core/DataArchive \
        Packages/Uintah/Core/Grid        \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/Labels      \
        Packages/Uintah/Core/Util        \
        Packages/Uintah/Core/Math        \
        Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/CCA/Ports        \
        Packages/Uintah/Core/ProblemSpec             \
        Packages/Uintah/CCA/Components/ProblemSpecification
endif

ifeq ($(IS_AIX),yes)
  LIBS := \
        $(TCL_LIBRARY) \
        $(TEEM_LIBRARY) \
        $(XML2_LIBRARY) \
        $(Z_LIBRARY) \
        $(THREAD_LIBRARY) \
        $(F_LIBRARY) \
        $(PETSC_LIBRARY) \
        $(HYPRE_LIBRARY) \
        $(BLAS_LIBRARY) \
        $(LAPACK_LIBRARY) \
        $(MPI_LIBRARY) \
        $(X_LIBRARY) \
        -lld $(M_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY) \
          $(CANTERA_LIBRARY) \
          $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) \
          $(MPI_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY)
endif

##############################################
# timeextract

SRCS    := $(SRCDIR)/timeextract.cc
PROGRAM := $(SRCDIR)/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# faceextract

SRCS    := $(SRCDIR)/faceextract.cc
PROGRAM := $(SRCDIR)/faceextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractV

SRCS    := $(SRCDIR)/extractV.cc
PROGRAM := $(SRCDIR)/extractV

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractF

SRCS    := $(SRCDIR)/extractF.cc
PROGRAM := $(SRCDIR)/extractF

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractS

SRCS    := $(SRCDIR)/extractS.cc
PROGRAM := $(SRCDIR)/extractS

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# partextract

SRCS    := $(SRCDIR)/partextract.cc
PROGRAM := $(SRCDIR)/partextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract

SRCS    := $(SRCDIR)/lineextract.cc
PROGRAM := $(SRCDIR)/lineextract

include $(SCIRUN_SCRIPTS)/program.mk
