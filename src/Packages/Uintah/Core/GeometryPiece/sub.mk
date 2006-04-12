# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/GeometryPiece

SRCS     += \
	$(SRCDIR)/GeometryObject.cc \
	$(SRCDIR)/GeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc \
	$(SRCDIR)/SphereMembraneGeometryPiece.cc \
	$(SRCDIR)/NaaBoxGeometryPiece.cc \
	$(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc \
	$(SRCDIR)/ConeGeometryPiece.cc \
	$(SRCDIR)/TriGeometryPiece.cc \
	$(SRCDIR)/UniformGrid.cc \
	$(SRCDIR)/UnionGeometryPiece.cc \
	$(SRCDIR)/DifferenceGeometryPiece.cc \
	$(SRCDIR)/IntersectionGeometryPiece.cc \
	$(SRCDIR)/FileGeometryPiece.cc \
	$(SRCDIR)/NullGeometryPiece.cc \
	$(SRCDIR)/GeometryPieceFactory.cc \
	$(SRCDIR)/ShellGeometryPiece.cc \
	$(SRCDIR)/ShellGeometryFactory.cc \
	$(SRCDIR)/CylinderShellPiece.cc \
	$(SRCDIR)/PlaneShellPiece.cc \
	$(SRCDIR)/SphereShellPiece.cc \
	$(SRCDIR)/GUVSphereShellPiece.cc \
	$(SRCDIR)/SmoothGeomPiece.cc \
	$(SRCDIR)/SmoothCylGeomPiece.cc \
	$(SRCDIR)/CorrugEdgeGeomPiece.cc

PSELIBS := \
	Core/Exceptions \
	Core/Geometry   \
	Core/Util       \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Util


LIBS       := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
