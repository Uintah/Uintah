# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Fracture

SRCS     += $(SRCDIR)/Fracture.cc \
	$(SRCDIR)/FractureFactory.cc \
	$(SRCDIR)/NormalFracture.cc \
	$(SRCDIR)/Lattice.cc \
	$(SRCDIR)/Cell.cc \
	$(SRCDIR)/ParticlesNeighbor.cc \
	$(SRCDIR)/CellsNeighbor.cc \
	$(SRCDIR)/DamagedParticle.cc \
	$(SRCDIR)/Equation.cc \
	$(SRCDIR)/LeastSquare.cc \
	$(SRCDIR)/CubicSpline.cc \
	$(SRCDIR)/QuarticSpline.cc \
	$(SRCDIR)/ExponentialSpline.cc \
	$(SRCDIR)/Visibility.cc \
	$(SRCDIR)/Connectivity.cc \
	$(SRCDIR)/CrackFace.cc \
	$(SRCDIR)/BoundaryBand.cc \
	$(SRCDIR)/SurfaceCouple.cc \
	$(SRCDIR)/SurfaceCouples.cc \
	$(SRCDIR)/IndexExchange.cc

PSELIBS	:= Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions


