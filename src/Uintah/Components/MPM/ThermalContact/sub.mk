#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/ThermalContact

SRCS     += $(SRCDIR)/ThermalContact.cc \
	$(SRCDIR)/ThermalContactFactory.cc

#
# $Log$
# Revision 1.2  2000/06/20 03:20:50  tan
# Added ThermalContactFactory class to interface with ProblemSpecification.
#
# Revision 1.1  2000/05/31 18:18:45  tan
# Create ThermalContact class to handle heat exchange in
# contact mechanics.
#
#
