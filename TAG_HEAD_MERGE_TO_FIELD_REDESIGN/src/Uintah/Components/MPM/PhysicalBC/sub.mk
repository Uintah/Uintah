#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/PhysicalBC

SRCS     += $(SRCDIR)/MPMPhysicalBCFactory.cc \
	$(SRCDIR)/ForceBC.cc

#
# $Log$
# Revision 1.1  2000/08/07 00:42:24  tan
# Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
# in MPM.  Currently implemented force boundary conditions.
#
#
