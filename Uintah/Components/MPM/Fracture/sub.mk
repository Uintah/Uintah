#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Fracture

SRCS     += $(SRCDIR)/Fracture.cc \
	$(SRCDIR)/FractureFactory.cc \
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
	$(SRCDIR)/LeastSquare.cc \
	$(SRCDIR)/BrokenCellShapeFunction.cc
	
	
#
# $Log$
# Revision 1.11  2000/09/06 18:54:06  tan
# Included BrokenCellShapeFunction.cc into sub.mk
#
# Revision 1.10  2000/07/06 05:01:13  tan
# Added Equation, LeastSquare, CubicSpline, QuarticSpline,ExponentialSpline,
# and LeastSquare classes.
#
# Revision 1.9  2000/06/06 02:50:10  tan
# Created class DamagedParticle to handle particle spliting when crack
# pass through.
#
# Revision 1.8  2000/06/06 00:07:26  tan
# Created class CellsNeighbor to handle cells neighbor computation.
#
# Revision 1.7  2000/06/05 21:20:03  tan
# Added class ParticlesNeighbor to handle neighbor particles searching.
#
# Revision 1.6  2000/06/05 20:50:56  tan
# A small modification.
#
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
