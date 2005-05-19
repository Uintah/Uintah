# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel

SRCS     += \
	$(SRCDIR)/RigidMaterial.cc        	\
	$(SRCDIR)/CompMooneyRivlin.cc        	\
	$(SRCDIR)/ConstitutiveModelFactory.cc 	\
	$(SRCDIR)/ConstitutiveModel.cc        	\
	$(SRCDIR)/ImplicitCM.cc 	       	\
	$(SRCDIR)/MPMMaterial.cc              	\
	$(SRCDIR)/CompNeoHook.cc              	\
	$(SRCDIR)/CNHDamage.cc              	\
	$(SRCDIR)/CNHPDamage.cc              	\
	$(SRCDIR)/CompNeoHookImplicit.cc 	\
	$(SRCDIR)/TransIsoHyper.cc              \
	$(SRCDIR)/TransIsoHyperImplicit.cc 	\
	$(SRCDIR)/ViscoTransIsoHyper.cc              \
	$(SRCDIR)/CompNeoHookPlas.cc          	\
	$(SRCDIR)/ViscoScram.cc               	\
	$(SRCDIR)/ViscoSCRAMHotSpot.cc               	\
	$(SRCDIR)/HypoElastic.cc              	\
	$(SRCDIR)/HypoElasticImplicit.cc      	\
	$(SRCDIR)/MWViscoElastic.cc           	\
	$(SRCDIR)/IdealGasMP.cc               	\
	$(SRCDIR)/Membrane.cc 			\
	$(SRCDIR)/ShellMaterial.cc 			\
	$(SRCDIR)/HypoElasticPlastic.cc \
	$(SRCDIR)/ElasticPlastic.cc 

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
