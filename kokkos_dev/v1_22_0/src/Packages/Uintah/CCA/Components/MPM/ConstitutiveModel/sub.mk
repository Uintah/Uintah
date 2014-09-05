# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel

SRCS     += \
	$(SRCDIR)/CompMooneyRivlin.cc        	\
	$(SRCDIR)/ConstitutiveModelFactory.cc 	\
	$(SRCDIR)/ConstitutiveModel.cc        	\
	$(SRCDIR)/MPMMaterial.cc              	\
	$(SRCDIR)/CompNeoHook.cc              	\
	$(SRCDIR)/CompNeoHookImplicit.cc 	\
	$(SRCDIR)/TransIsoHyper.cc              \
	$(SRCDIR)/TransIsoHyperImplicit.cc 	\
	$(SRCDIR)/CompNeoHookPlas.cc          	\
	$(SRCDIR)/ViscoScram.cc               	\
	$(SRCDIR)/ViscoScramForBinder.cc      	\
	$(SRCDIR)/HypoElastic.cc              	\
	$(SRCDIR)/HypoElasticImplicit.cc      	\
	$(SRCDIR)/MWViscoElastic.cc           	\
	$(SRCDIR)/IdealGasMP.cc               	\
	$(SRCDIR)/Membrane.cc 			\
	$(SRCDIR)/ShellMaterial.cc 			\
	$(SRCDIR)/GUVMaterial.cc 			\
	$(SRCDIR)/HypoElasticPlastic.cc \
	$(SRCDIR)/HyperElasticPlastic.cc \
	$(SRCDIR)/DamageModel.cc \
	$(SRCDIR)/DamageModelFactory.cc \
	$(SRCDIR)/JohnsonCookDamage.cc \
	$(SRCDIR)/HancockMacKenzieDamage.cc \
	$(SRCDIR)/MPMEquationOfState.cc \
	$(SRCDIR)/MPMEquationOfStateFactory.cc \
	$(SRCDIR)/DefaultHypoElasticEOS.cc \
	$(SRCDIR)/DefaultHyperElasticEOS.cc \
	$(SRCDIR)/MieGruneisenEOS.cc \
	$(SRCDIR)/PlasticityModel.cc \
	$(SRCDIR)/PlasticityModelFactory.cc \
	$(SRCDIR)/IsoHardeningPlastic.cc \
	$(SRCDIR)/JohnsonCookPlastic.cc \
	$(SRCDIR)/MTSPlastic.cc \
	$(SRCDIR)/SCGPlastic.cc \
	$(SRCDIR)/YieldCondition.cc \
	$(SRCDIR)/YieldConditionFactory.cc \
	$(SRCDIR)/GursonYield.cc \
	$(SRCDIR)/RousselierYield.cc \
	$(SRCDIR)/VonMisesYield.cc \
	$(SRCDIR)/StabilityCheck.cc \
	$(SRCDIR)/StabilityCheckFactory.cc \
	$(SRCDIR)/AcousticTensorCheck.cc \
	$(SRCDIR)/BeckerCheck.cc \
	$(SRCDIR)/DruckerCheck.cc \
	$(SRCDIR)/DruckerBeckerCheck.cc \
	$(SRCDIR)/PlasticityState.cc 

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
