#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/Writers

SRCS     += $(SRCDIR)/MPWriter.cc

PSELIBS := Uintah/Datatypes/Particles PSECore/Dataflow \
	SCICore/TclInterface SCICore/Exceptions SCICore/Persistent \
	SCICore/Containers SCICore/Thread SCICore/Datatypes
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:30:19  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
