# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/fortran

SRCS     += \
	$(SRCDIR)/Hooke.F	\
	$(SRCDIR)/MIGUtilsF.F	\
	$(SRCDIR)/MIGUtils.cc

#$(SRCDIR)/HookeChk.$(OBJEXT): $(SRCDIR)/HookeChk_fort.h
