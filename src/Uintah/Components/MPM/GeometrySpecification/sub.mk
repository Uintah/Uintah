#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/GeometrySpecification

SRCS     += $(SRCDIR)/GeometryGrid.cc $(SRCDIR)/GeometryObject.cc \
	$(SRCDIR)/GeometryPiece.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/Problem.cc $(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc $(SRCDIR)/TriGeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc 

#
# $Log$
# Revision 1.2  2000/04/14 02:05:47  jas
# Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
# Tri.  This made the GeometryObject class simpler since many of the
# methods are now relegated to the GeometryPiece subclasses.
#
# Revision 1.1  2000/03/17 09:29:40  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
