#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/ICE/EOS

SRCS     += $(SRCDIR)/EquationOfState.cc \
	$(SRCDIR)/EquationOfStateFactory.cc \
	$(SRCDIR)/IdealGas.cc 

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil Uintah/Math
LIBS	:= $(XML_LIBRARY)        

#
# $Log$
# Revision 1.1.2.1  2000/10/19 05:17:42  sparker
# Merge changes from main branch into csafe_risky1
#
# Revision 1.1  2000/10/06 04:02:16  jas
# Move into a separate EOS directory.
#

