#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Fracture

SRCS     += $(SRCDIR)/Fracture.cc \
	$(SRCDIR)/FractureFactory.cc

#
# $Log$
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
