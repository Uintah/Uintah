# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/PartVel.cc \
  $(SRCDIR)/Devolatilization.cc \
  $(SRCDIR)/CharOxidation.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/YamamotoDevol.cc \
  $(SRCDIR)/CharOxidationShaddix.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/SimpleHeatTransfer.cc \
  $(SRCDIR)/ShaddixHeatTransfer.cc \
  $(SRCDIR)/XDragModel.cc \
  $(SRCDIR)/YDragModel.cc \
  $(SRCDIR)/ZDragModel.cc \
  $(SRCDIR)/DragModel.cc

$(SRCDIR)/DragModel.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
