# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/Size.cc \
  $(SRCDIR)/ParticleDensity.cc \
  $(SRCDIR)/ConstantSizeCoal.cc \
  $(SRCDIR)/ConstantDensityCoal.cc \
  $(SRCDIR)/ConstantSizeInert.cc \
  $(SRCDIR)/ConstantDensityInert.cc \
  $(SRCDIR)/ParticleVelocity.cc \
  $(SRCDIR)/Balachandar.cc \
  $(SRCDIR)/Devolatilization.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/CharOxidation.cc \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/CoalParticleHeatTransfer.cc \
  $(SRCDIR)/DragModel.cc \


  #$(SRCDIR)/InertParticleHeatTransfer.cc \

