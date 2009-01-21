# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/fortran

FFLAGS += -DNOEOSMOD

SRCS     += \
	$(SRCDIR)/Hooke.F	\
	$(SRCDIR)/geochk.F	\
	$(SRCDIR)/Isotropic_Geomaterial_init.F	\
	$(SRCDIR)/Isotropic_Geomaterial_calcs.F	\
	$(SRCDIR)/MIGUtilsF.F	\
	$(SRCDIR)/MIGUtils.cc

#$(SRCDIR)/HookeChk.$(OBJEXT): $(SRCDIR)/HookeChk_fort.h
