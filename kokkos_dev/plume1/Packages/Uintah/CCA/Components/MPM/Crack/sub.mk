# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Crack

SRCS     += \
	$(SRCDIR)/ReadAndDiscretizeCracks.cc \
	$(SRCDIR)/ParticleNodePairVelocityField.cc \
	$(SRCDIR)/CrackSurfaceContact.cc \
	$(SRCDIR)/FractureParametersCalculation.cc \
	$(SRCDIR)/CrackPropagation.cc \
	$(SRCDIR)/MoveCracks.cc \
	$(SRCDIR)/UpdateCrackFront.cc 

