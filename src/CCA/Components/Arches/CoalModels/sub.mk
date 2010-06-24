# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/Size.cc \
  $(SRCDIR)/ParticleDensity.cc \
  $(SRCDIR)/ConstantSize.cc \
  $(SRCDIR)/ParticleVelocity.cc \
  $(SRCDIR)/Balachandar.cc \
  $(SRCDIR)/Devolatilization.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/CharOxidation.cc \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/SimpleHeatTransfer.cc \
  $(SRCDIR)/DragModel.cc \

