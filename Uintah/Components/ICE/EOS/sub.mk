#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/ICE/EOS

SRCS     += $(SRCDIR)/EquationOfState.cc \
	$(SRCDIR)/EquationOfStateFactory.cc \
	$(SRCDIR)/IdealGas.cc $(SRCDIR)/Harlow.cc \
	$(SRCDIR)/StiffGas.cc

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil Uintah/Math
LIBS	:= $(XML_LIBRARY)        

#
# $Log$
# Revision 1.3  2000/10/31 04:14:28  jas
# Added stiff gas EOS type.  It is just a copy of IdealGas.
#
# Revision 1.2  2000/10/26 23:37:05  jas
# Added Harlow EOS.
#
# Revision 1.1  2000/10/06 04:02:16  jas
# Move into a separate EOS directory.
#

