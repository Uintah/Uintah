#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/EGI

SRCS     += $(SRCDIR)/Anneal.cc $(SRCDIR)/DipoleInSphere.cc

PSELIBS := PSECore/Datatypes PSECore/Dataflow SCICore/Containers \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/TclInterface SCICore/Datatypes
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:07  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:39  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
