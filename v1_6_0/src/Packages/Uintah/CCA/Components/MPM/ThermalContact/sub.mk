# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ThermalContact

SRCS     += $(SRCDIR)/ThermalContact.cc \
	$(SRCDIR)/STThermalContact.cc \
	$(SRCDIR)/NullThermalContact.cc \
	$(SRCDIR)/ThermalContactFactory.cc

