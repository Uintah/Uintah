# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel

SRCS     += \
	$(SRCDIR)/CompMooneyRivlin.cc        	\
	$(SRCDIR)/ConstitutiveModelFactory.cc 	\
	$(SRCDIR)/ConstitutiveModel.cc        	\
	$(SRCDIR)/MPMMaterial.cc              	\
	$(SRCDIR)/CompNeoHook.cc              	\
	$(SRCDIR)/CompNeoHookImplicit.cc 	\
	$(SRCDIR)/CompNeoHookPlas.cc          	\
	$(SRCDIR)/ViscoScram.cc               	\
	$(SRCDIR)/ViscoScramForBinder.cc      	\
	$(SRCDIR)/HypoElastic.cc              	\
	$(SRCDIR)/MWViscoElastic.cc           	\
	$(SRCDIR)/IdealGasMP.cc               	\
	$(SRCDIR)/Membrane.cc 			\
	$(SRCDIR)/DamageModel.cc \
	$(SRCDIR)/DamageModelFactory.cc \
	$(SRCDIR)/DefaultHypoElasticEOS.cc \
	$(SRCDIR)/DefaultHyperElasticEOS.cc \
	$(SRCDIR)/MPMEquationOfState.cc \
	$(SRCDIR)/MPMEquationOfStateFactory.cc \
	$(SRCDIR)/HypoElasticPlastic.cc \
	$(SRCDIR)/HyperElasticPlastic.cc \
	$(SRCDIR)/IsoHardeningPlastic.cc \
	$(SRCDIR)/JohnsonCookDamage.cc \
	$(SRCDIR)/JohnsonCookPlastic.cc \
	$(SRCDIR)/MieGruneisenEOS.cc \
	$(SRCDIR)/MTSPlastic.cc \
	$(SRCDIR)/PlasticityModel.cc \
	$(SRCDIR)/PlasticityModelFactory.cc 

PSELIBS := Packages/Uintah/Core/Grid \
	Packages/Uintah/CCA/Components/ICE \
	Packages/Uintah/CCA/Components/HETransformation \
	Core/Datatypes \
	Core/Util

