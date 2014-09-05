# 
# 
# The MIT License
# 
# Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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
#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Arches/Radiation

SRCS += \
        $(SRCDIR)/RadiationModel.cc   \
        $(SRCDIR)/RadiationSolver.cc  \
        $(SRCDIR)/DORadiationModel.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/RadLinearSolver.cc
else
  SRCS += $(SRCDIR)/FakeRadLinearSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/RadHypreSolver.cc
endif

PSELIBS := \
        CCA/Components/Arches/Radiation/fortran \
        Core/ProblemSpec   \
        Core/Grid          \
        Core/Util          \
        Core/Disclosure    \
        Core/Exceptions    \
        Core/Math          \
        Core/Exceptions \
        Core/Util       \
        Core/Thread     \
        Core/Geometry   

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY) \
        $(LAPACK_LIBRARY) $(BLAS_LIBRARY) $(THREAD_LIBRARY)

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordr_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordrss_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordrtn_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radarray_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radcal_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radcoef_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radwsgg_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdombc_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomsolve_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomsrc_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomflux_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdombmcalc_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomvolq_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rshsolve_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rshresults_fort.h



