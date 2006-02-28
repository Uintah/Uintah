#
# Makefile fragment for this subdirectory
# $Id: sub.mk,v 1.12 2005/03/02 20:19:24 worthen Exp $
#

SRCDIR   := Packages/Uintah/CCA/Components/Models/Radiation

SUBDIRS := $(SRCDIR)/fortran
include $(SCIRUN_SCRIPTS)/recurse.mk

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

$(SRCDIR)/Models_CellInformation.o: $(SRCDIR)/fortran/m_cellg_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rordr_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rordrss_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rordrtn_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_radarray_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_radcal_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_radcoef_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_radwsgg_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdombc_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdomsolve_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdomsrc_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdomflux_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdombmcalc_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rdomvolq_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rshsolve_fort.h
$(SRCDIR)/Models_DORadiationModel.o: $(SRCDIR)/fortran/m_rshresults_fort.h
