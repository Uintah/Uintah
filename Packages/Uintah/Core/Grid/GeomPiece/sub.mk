# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/Core/Grid/GeomPiece

SRCS     += \
	$(SRCDIR)/GeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc \
	$(SRCDIR)/SphereMembraneGeometryPiece.cc \
	$(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc \
	$(SRCDIR)/ConeGeometryPiece.cc \
	$(SRCDIR)/TriGeometryPiece.cc \
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
	$(SRCDIR)/SphereShellPiece.cc 

PSELIBS := \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

