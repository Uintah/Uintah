#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/GeometrySpecification

SRCS     += $(SRCDIR)/GeometryGrid.cc $(SRCDIR)/GeometryObject.cc \
	$(SRCDIR)/GeometryPiece.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/Problem.cc

#
# $Log$
# Revision 1.1  2000/03/17 09:29:40  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
