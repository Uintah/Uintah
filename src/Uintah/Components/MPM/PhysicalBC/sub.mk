#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/PhysicalBC

SRCS     += $(SRCDIR)/MPMPhysicalBCFactory.cc \
	$(SRCDIR)/ForceBC.cc \
	$(SRCDIR)/CrackBC.cc

#
# $Log$
# Revision 1.2  2000/12/30 05:08:12  tan
# Fixed a problem concerning patch and ghost in fracture computations.
#
# Revision 1.1  2000/08/07 00:42:24  tan
# Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
# in MPM.  Currently implemented force boundary conditions.
#
#
