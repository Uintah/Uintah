# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/ICE/Thermo

SRCS     += $(SRCDIR)/ThermoInterface.cc \
	$(SRCDIR)/ThermoFactory.cc \
	$(SRCDIR)/ConstantThermo.cc \
        $(SRCDIR)/CanteraDetailed.cc \
        $(SRCDIR)/CanteraSingleMixture.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 

# Uncomment this like to compile with cantera
CANTERA_DIR := /Users/sparker/sw
ifneq ($(CANTERA_DIR),)
 INCLUDES := $(INCLUDES) -I$(CANTERA_DIR)/include
 CANTERA_LIBRARY := -L$(CANTERA_DIR)/lib/1.6.0 -loneD -lzeroD -ltransport -lcantera -lrecipes -lcvode -lctmath -ltpx -lconverters -lctcxx
endif
