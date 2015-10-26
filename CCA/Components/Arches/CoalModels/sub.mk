# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/PartVel.cc \
  $(SRCDIR)/Devolatilization.cc \
  $(SRCDIR)/CharOxidation.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/RichardsFletcherDevol.cc \
  $(SRCDIR)/FOWYDevol.cc \
  $(SRCDIR)/YamamotoDevol.cc \
  $(SRCDIR)/CharOxidationShaddix.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/EnthalpyShaddix.cc \
  $(SRCDIR)/MaximumTemperature.cc \
	$(SRCDIR)/SimpleBirth.cc \
	$(SRCDIR)/ParticleConvection.cc \
	$(SRCDIR)/Thermophoresis.cc \
	$(SRCDIR)/Deposition.cc \
  $(SRCDIR)/DragModel.cc 

$(SRCDIR)/ShaddixHeatTransfer.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
$(SRCDIR)/EnthalpyShaddix.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
