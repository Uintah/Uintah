#
# Makefile fragment for this subdirectory
# $Id: sub.mk,v 1.12 2005/03/02 20:19:24 worthen Exp $
#

SRCDIR   := Packages/Uintah/CCA/Components/Models/Radiation

SUBDIRS := $(SRCDIR)/fortran
include $(SCIRUN_SCRIPTS)/recurse.mk

SRCS += \
       $(SRCDIR)/CellInformation.cc \
       $(SRCDIR)/DORadiationModel.cc \
       $(SRCDIR)/RadiationConstVariables.cc \
       $(SRCDIR)/RadiationDriver.cc \
       $(SRCDIR)/RadiationModel.cc \
       $(SRCDIR)/RadiationSolver.cc \
       $(SRCDIR)/RadiationVariables.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/RadLinearSolver.cc
else
  SRCS += $(SRCDIR)/FakeRadLinearSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/RadHypreSolver.cc
endif

$(SRCDIR)/CellInformation.o: $(SRCDIR)/fortran/cellg_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rordr_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rordrss_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rordrtn_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/radarray_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/radcal_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/radcoef_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/radwsgg_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdombc_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdomsolve_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdomsrc_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdomflux_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdombmcalc_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rdomvolq_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rshsolve_fort.h
$(SRCDIR)/DORadiationModel.o: $(SRCDIR)/fortran/rshresults_fort.h
