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
  $(SRCDIR)/BTDevol.cc \
  $(SRCDIR)/YamamotoDevol.cc \
  $(SRCDIR)/CharOxidationShaddix.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/EnthalpyShaddix.cc \
	$(SRCDIR)/SimpleBirth.cc \
	$(SRCDIR)/ParticleConvection.cc \
  $(SRCDIR)/DragModel.cc 

$(SRCDIR)/ShaddixHeatTransfer.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
$(SRCDIR)/EnthalpyShaddix.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
