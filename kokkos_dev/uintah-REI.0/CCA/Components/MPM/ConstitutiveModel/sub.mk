# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel

SRCS     += \
	$(SRCDIR)/ConstitutiveModelFactory.cc 	\
	$(SRCDIR)/ConstitutiveModel.cc        	\
	$(SRCDIR)/ImplicitCM.cc 	       	\
	$(SRCDIR)/MPMMaterial.cc              	\
        $(SRCDIR)/ShellMaterial.cc              \
	$(SRCDIR)/CompNeoHookPlas.cc            \
	$(SRCDIR)/CompNeoHook.cc                \
	$(SRCDIR)/CompNeoHookImplicit.cc        \
	$(SRCDIR)/HypoElastic.cc                \
	$(SRCDIR)/SoilFoam.cc

SUBDIRS := $(SRCDIR)/PlasticityModels

include $(SCIRUN_SCRIPTS)/recurse.mk

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
