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


SRCDIR := CCA/Components/Arches/Radiation

SRCS += \
        $(SRCDIR)/RadiationSolver.cc  \
        $(SRCDIR)/DORadiationModel.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/RadPetscSolver.cc
else
  SRCS += $(SRCDIR)/FakeRadPetscSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/RadHypreSolver.cc
endif

$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordr_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordrss_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rordrtn_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radarray_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radcal_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radcoef_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/radwsgg_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomsolve_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomsrc_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomflux_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdombmcalc_fort.h
$(SRCDIR)/DORadiationModel.$(OBJEXT): $(SRCDIR)/fortran/rdomvolq_fort.h



