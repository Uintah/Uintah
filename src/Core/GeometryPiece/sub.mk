#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the \"Software\"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/GeometryPiece

SRCS     += \
	$(SRCDIR)/GeometryObject.cc \
	$(SRCDIR)/GeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc \
	$(SRCDIR)/SphereMembraneGeometryPiece.cc \
	$(SRCDIR)/NaaBoxGeometryPiece.cc \
	$(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc \
	$(SRCDIR)/TorusGeometryPiece.cc \
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
	$(SRCDIR)/SmoothGeomPiece.cc \
	$(SRCDIR)/SmoothCylGeomPiece.cc \
	$(SRCDIR)/CorrugEdgeGeomPiece.cc \
        $(SRCDIR)/EllipsoidGeometryPiece.cc 
#	$(SRCDIR)/GUVSphereShellPiece.cc \

PSELIBS := \
	Core/Exceptions \
	Core/Geometry   \
	Core/Util       \
	Core/Exceptions  \
	Core/Grid        \
	Core/Math        \
	Core/Parallel    \
	Core/ProblemSpec \



LIBS       := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
