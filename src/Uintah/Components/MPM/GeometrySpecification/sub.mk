#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/GeometrySpecification

SRCS     += $(SRCDIR)/GeometryGrid.cc $(SRCDIR)/GeometryObject.cc \
	$(SRCDIR)/GeometryPiece.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/Problem.cc $(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc $(SRCDIR)/TriGeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc  $(SRCDIR)/BoxGeometryObject.cc \
	$(SRCDIR)/SphereGeometryObject.cc \
	$(SRCDIR)/CylinderGeometryObject.cc \
	$(SRCDIR)/TriGeometryObject.cc \
	$(SRCDIR)/UnionGeometryObject.cc \
	$(SRCDIR)/DifferenceGeometryObject.cc \
	$(SRCDIR)/IntersectionGeometryObject.cc \
	$(SRCDIR)/GeometryObjectFactory.cc


#
# $Log$
# Revision 1.4  2000/04/20 15:09:27  jas
# Added factory methods for GeometryObjects.
#
# Revision 1.3  2000/04/19 21:31:09  jas
# Revamping of the way objects are defined.  The different geometry object
# subtypes only do a few simple things such as testing whether a point
# falls inside the object and also gets the bounding box for the object.
# The constructive solid geometry objects:union,difference, and intersection
# have the same simple operations.
#
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
