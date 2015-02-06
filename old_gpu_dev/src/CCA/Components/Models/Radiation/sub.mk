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
#
# Makefile fragment for this subdirectory
# $Id: sub.mk,v 1.12 2005/03/02 20:19:24 worthen Exp $
#

SRCDIR   := CCA/Components/Models/Radiation

SUBDIRS := $(SRCDIR)/RMCRT
           
ifneq ($(NO_FORTRAN),yes)
  SRCS += \
       $(SRCDIR)/Models_CellInformation.cc \
       $(SRCDIR)/Models_DORadiationModel.cc \
       $(SRCDIR)/RadiationConstVariables.cc \
       $(SRCDIR)/RadiationDriver.cc \
       $(SRCDIR)/Models_RadiationModel.cc \
       $(SRCDIR)/Models_RadiationSolver.cc \
       $(SRCDIR)/RadiationVariables.cc


  ifeq ($(HAVE_PETSC),yes)
    SRCS += $(SRCDIR)/Models_PetscSolver.cc
  else
    SRCS += $(SRCDIR)/Models_FakePetscSolver.cc
  endif

  ifeq ($(HAVE_HYPRE),yes)
    SRCS += $(SRCDIR)/Models_HypreSolver.cc
  endif

  SUBDIRS += $(SRCDIR)/fortran
  
  $(SRCDIR)/Models_CellInformation.$(OBJEXT): $(SRCDIR)/fortran/m_cellg_fort.h

  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rordr_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rordrtn_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_radarray_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_radcal_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_radcoef_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_radwsgg_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdombc_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdomsolve_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdomsrc_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdomflux_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdombmcalc_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rdomvolq_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rshsolve_fort.h
  $(SRCDIR)/Models_DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/m_rshresults_fort.h
  
endif

include $(SCIRUN_SCRIPTS)/recurse.mk

