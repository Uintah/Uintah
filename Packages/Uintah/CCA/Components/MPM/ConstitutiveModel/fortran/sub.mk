# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/fortran

SRCS     += \
	$(SRCDIR)/Hooke.F	\
	$(SRCDIR)/Helper.F

PSELIBS := Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/ICE \
	Core/Datatypes \
	Core/Exceptions \
	Core/Geometry \
	Core/Math \
	Core/Util

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#$(SRCDIR)/HookeChk.$(OBJEXT): $(SRCDIR)/HookeChk_fort.h

