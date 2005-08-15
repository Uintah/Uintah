# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Models/test

# Uncomment this like to compile with cantera
#CANTERA_DIR := /home/sci/sparker/canterataz
ifneq ($(CANTERA_DIR),)
 INCLUDES := $(INCLUDES) -I$(CANTERA_DIR)/include
 CANTERA_LIBRARY := -L$(CANTERA_DIR)/lib/cantera -loneD -lzeroD -ltransport -lconverters -lcantera -lrecipes -lcvode -lctlapack -lctmath -lctblas -lctcxx
endif

SRCS	+= \
       $(SRCDIR)/Mixing.cc \
       $(SRCDIR)/ArchesTable.cc \
       $(SRCDIR)/MaterialProperties.cc \
       $(SRCDIR)/PassiveScalar.cc \
       $(SRCDIR)/TableInterface.cc \
       $(SRCDIR)/TableFactory.cc \
       $(SRCDIR)/TestModel.cc \
       $(SRCDIR)/VorticityConfinement.cc \
       $(SRCDIR)/flameSheet_rxn.cc 
#       $(SRCDIR)/AdiabaticTable.cc \
#       $(SRCDIR)/NonAdiabaticTable.cc \
#       $(SRCDIR)/SimpleRxn.cc \
#       $(SRCDIR)/Mixing2.cc
#       $(SRCDIR)/Mixing2.cc \
#       $(SRCDIR)/Mixing3.cc
