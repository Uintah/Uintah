# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/CCA/Components/MPM/ConstitutiveModel

SRCS   += \
        $(SRCDIR)/RigidMaterial.cc              \
        $(SRCDIR)/CompMooneyRivlin.cc           \
        $(SRCDIR)/ConstitutiveModelFactory.cc   \
        $(SRCDIR)/ConstitutiveModel.cc          \
        $(SRCDIR)/ImplicitCM.cc                 \
        $(SRCDIR)/MPMMaterial.cc                \
        $(SRCDIR)/CompNeoHook.cc                \
        $(SRCDIR)/CNHDamage.cc                  \
        $(SRCDIR)/CNHPDamage.cc                 \
        $(SRCDIR)/CompNeoHookImplicit.cc        \
        $(SRCDIR)/TransIsoHyper.cc              \
        $(SRCDIR)/TransIsoHyperImplicit.cc      \
        $(SRCDIR)/ViscoTransIsoHyper.cc         \
        $(SRCDIR)/ViscoTransIsoHyperImplicit.cc \
        $(SRCDIR)/CompNeoHookPlas.cc            \
        $(SRCDIR)/ViscoScram.cc                 \
        $(SRCDIR)/ViscoSCRAMHotSpot.cc          \
        $(SRCDIR)/HypoElastic.cc                \
        $(SRCDIR)/HypoElasticFortran.cc         \
        $(SRCDIR)/HypoElasticImplicit.cc        \
        $(SRCDIR)/ViscoScramImplicit.cc         \
        $(SRCDIR)/MWViscoElastic.cc             \
        $(SRCDIR)/IdealGasMP.cc                 \
        $(SRCDIR)/Membrane.cc                   \
        $(SRCDIR)/ShellMaterial.cc              \
        $(SRCDIR)/HypoElasticPlastic.cc         \
        $(SRCDIR)/ElasticPlastic.cc             \
        $(SRCDIR)/ElasticPlasticHP.cc           \
        $(SRCDIR)/SmallStrainPlastic.cc         \
        $(SRCDIR)/Water.cc                      \
        $(SRCDIR)/ViscoPlastic.cc               \
        $(SRCDIR)/Kayenta.cc                    \
        $(SRCDIR)/SoilFoam.cc

SUBDIRS := \
        $(SRCDIR)/PlasticityModels \
        $(SRCDIR)/fortran

include $(SCIRUN_SCRIPTS)/recurse.mk
