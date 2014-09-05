# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/CoalModels

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc \
  $(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/PartVel.cc \
  $(SRCDIR)/Devolatilization.cc \
  $(SRCDIR)/KobayashiSarofimDevol.cc \
  $(SRCDIR)/ConstantModel.cc  \
  $(SRCDIR)/HeatTransfer.cc \
  $(SRCDIR)/SimpleHeatTransfer.cc \
  $(SRCDIR)/XDragModel.cc \
  $(SRCDIR)/YDragModel.cc \
  $(SRCDIR)/ZDragModel.cc \
  $(SRCDIR)/DragModel.cc
