#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/so_prologue.mk

SRCDIR := Phil

SUBDIRS := $(SRCDIR)/Modules
include $(SRCTOP)/scripts/recurse.mk

ifeq ($(LARGESOS),yes)
PSELIBS := PSECore SCICore
else
PSELIBS := PSECore/Datatypes PSECore/Dataflow SCICore/Persistent \
	SCICore/Exceptions SCICore/Containers SCICore/Geometry \
	SCICore/Geom SCICore/Thread SCICore/TclInterface \
	SCICore/Datatypes
endif
LIBS := $(GL_LIBS) -lm $(THREAD_LIBS)

include $(SRCTOP)/scripts/so_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:27  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:06  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
