#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Geometry

SRCS     += $(SRCDIR)/BBox.cc $(SRCDIR)/Grid.cc $(SRCDIR)/IntVector.cc \
	$(SRCDIR)/Point.cc \
	$(SRCDIR)/Transform.cc $(SRCDIR)/Plane.cc $(SRCDIR)/Vector.cc \
	$(SRCDIR)/Ray.cc  $(SRCDIR)/Quaternion.cc

PSELIBS := SCICore/Containers SCICore/Exceptions SCICore/Tester
LIBS := $(DEFAULT_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/08/02 09:10:30  samsonov
# Added Quaternion.cc path
#
# Revision 1.3  2000/04/13 06:48:39  sparker
# Implemented more of IntVector class
#
# Revision 1.2  2000/03/20 19:37:41  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:28  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
