#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Contact

SRCS     += $(SRCDIR)/NullContact.cc $(SRCDIR)/SingleVelContact.cc

#
# $Log$
# Revision 1.2  2000/03/21 02:14:48  dav
# updated SingleVel to SingleVelContact
#
# Revision 1.1  2000/03/17 09:29:36  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
