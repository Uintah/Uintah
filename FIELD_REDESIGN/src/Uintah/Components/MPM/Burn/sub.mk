#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Burn

SRCS     += $(SRCDIR)/NullHEBurn.cc $(SRCDIR)/SimpleHEBurn.cc \
	$(SRCDIR)/HEBurnFactory.cc  $(SRCDIR)/HEBurn.cc

#
# $Log$
# Revision 1.1  2000/06/02 22:48:26  jas
# Added infrastructure for Burn models.
#
