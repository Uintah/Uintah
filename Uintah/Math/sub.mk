#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Math

SRCS     += $(SRCDIR)/Primes.cc	$(SRCDIR)/CubicPolyRoots.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/08/15 19:09:56  witzel
# Included CubicPolyRoots stuff.
#
# Revision 1.2  2000/03/20 19:38:39  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:06  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
