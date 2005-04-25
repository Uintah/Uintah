# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Examples

SRCS     += \
	$(SRCDIR)/AMRWave.cc \
	$(SRCDIR)/Wave.cc \
	$(SRCDIR)/Poisson1.cc \
	$(SRCDIR)/Poisson2.cc \
	$(SRCDIR)/Burger.cc \
	$(SRCDIR)/Poisson3.cc \
	$(SRCDIR)/ParticleTest1.cc \
	$(SRCDIR)/SimpleCFD.cc \
	$(SRCDIR)/AMRSimpleCFD.cc \
	$(SRCDIR)/Interpolator.cc \
	$(SRCDIR)/ExamplesLabel.cc \
	$(SRCDIR)/RegridderTest.cc \
	$(SRCDIR)/SolverTest1.cc \
	$(SRCDIR)/BoundaryConditions.cc \
	$(SRCDIR)/RegionDB.cc

PSELIBS := \
	Core/Geometry \
	Core/Util \
	Core/Exceptions                   \
	Packages/Uintah/CCA/Ports         \
	Packages/Uintah/Core/Grid         \
	Packages/Uintah/Core/Util         \
	Packages/Uintah/Core/ProblemSpec  \
	Packages/Uintah/Core/GeometryPiece  \
	Packages/Uintah/Core/Exceptions   \
	Packages/Uintah/Core/Disclosure   \
	Packages/Uintah/Core/Math         \
	Packages/Uintah/Core/Parallel

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

