#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/Path

SRCS     += $(SRCDIR)/EditPath.cc

PSELIBS := DaveW/Datatypes/General PSECore/Dataflow PSECore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Containers \
	SCICore/Math SCICore/Thread SCICore/Geom SCICore/TclInterface
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:25:54  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
