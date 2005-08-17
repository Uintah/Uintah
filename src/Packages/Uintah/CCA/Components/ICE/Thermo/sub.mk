# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/ICE/Thermo

SRCS     += $(SRCDIR)/ThermoInterface.cc \
	$(SRCDIR)/ThermoFactory.cc \
	$(SRCDIR)/ConstantThermo.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


