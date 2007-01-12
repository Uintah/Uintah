# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Crack

SRCS     += \
	$(SRCDIR)/ReadAndDiscretizeCracks.cc \
	$(SRCDIR)/CrackGeometryFactory.cc \
	$(SRCDIR)/CrackGeometry.cc \
	$(SRCDIR)/NullCrack.cc \
	$(SRCDIR)/QuadCrack.cc \
	$(SRCDIR)/CurvedQuadCrack.cc \
	$(SRCDIR)/TriangularCrack.cc \
	$(SRCDIR)/ArcCrack.cc \
	$(SRCDIR)/EllipticCrack.cc \
	$(SRCDIR)/PartialEllipticCrack.cc \
	$(SRCDIR)/ParticleNodePairVelocityField.cc \
	$(SRCDIR)/CrackSurfaceContact.cc \
	$(SRCDIR)/FractureParametersCalculation.cc \
	$(SRCDIR)/CrackPropagation.cc \
	$(SRCDIR)/MoveCracks.cc \
	$(SRCDIR)/UpdateCrackFront.cc 

