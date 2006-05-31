# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/PhysicalBC

SRCS     += \
	$(SRCDIR)/MPMPhysicalBCFactory.cc \
	$(SRCDIR)/ForceBC.cc              \
	$(SRCDIR)/NormalForceBC.cc              \
	$(SRCDIR)/PressureBC.cc              \
	$(SRCDIR)/CrackBC.cc

