#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Example

SRCS     += $(SRCDIR)/GeomPortTest.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Containers \
	SCICore/Thread SCICore/Geom SCICore/TclInterface
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:54  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:49  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
