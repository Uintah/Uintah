#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/Writers

SRCS     += $(SRCDIR)/MPWriter.cc

PSELIBS := Uintah/Datatypes PSECore/Dataflow \
	SCICore/TclInterface SCICore/Exceptions SCICore/Persistent \
	SCICore/Containers SCICore/Thread SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/20 17:57:20  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.2  2000/03/20 19:38:45  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:19  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
