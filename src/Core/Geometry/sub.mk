#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Geometry

SRCS     += $(SRCDIR)/BBox.cc $(SRCDIR)/Grid.cc $(SRCDIR)/Point.cc \
	$(SRCDIR)/Transform.cc $(SRCDIR)/Plane.cc $(SRCDIR)/Vector.cc \
	$(SRCDIR)/Ray.cc

PSELIBS := SCICore/Containers SCICore/Exceptions SCICore/Tester
LIBS := $(DEFAULT_LIBS) -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:28:28  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
