#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Iterators

SRCS     += $(SRCDIR)/MeshIterator.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Persistent \
	SCICore/Exceptions SCICore/Thread SCICore/TclInterface \
	SCICore/Containers SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:58  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
