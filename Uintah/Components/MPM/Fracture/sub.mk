#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Fracture

SRCS     += $(SRCDIR)/Fracture.cc \
	$(SRCDIR)/FractureFactory.cc \
	$(SRCDIR)/Lattice.cc
	$(SRCDIR)/Cell.cc

#
# $Log$
# Revision 1.5  2000/06/05 19:46:03  tan
# Cell class will be designed to carray a link list of particle indexes
# in a cell.  This will facilitate the seaching of particles from a given
# cell.
#
# Revision 1.4  2000/06/05 17:22:20  tan
# Lattice class will be designed to make it easier to handle the grid/particle
# relationship in a given patch and a given velocity field.
#
# Revision 1.3  2000/05/10 17:21:46  tan
# Added FractureFactory.cc.
#
# Revision 1.2  2000/05/05 00:05:50  tan
# Added Fracture.cc.
#
# Revision 1.1  2000/03/17 09:29:38  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
