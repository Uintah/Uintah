#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/HeatConduction

SRCS     += $(SRCDIR)/HeatConduction.cc \
	$(SRCDIR)/HeatConductionFactory.cc

# $Log$
# Revision 1.1  2000/06/20 17:59:52  tan
# Heat Conduction model created to move heat conduction part of code from MPM.
# Thus make MPM clean and easy to maintain.
#
