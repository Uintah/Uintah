#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/so_prologue.mk

SRCDIR := Kurt

SUBDIRS := $(SRCDIR)/GUI $(SRCDIR)/Geom $(SRCDIR)/Modules

include $(OBJTOP_ABS)/scripts/recurse.mk

ifeq ($(LARGESOS),yes)
PSELIBS := PSECore SCICore
else
PSELIBS := PSECore/Datatypes PSECore/Dataflow SCICore/Containers \
	SCICore/TclInterface SCICore/Exceptions SCICore/Persistent \
	SCICore/Thread SCICore/Geom SCICore/Datatypes SCICore/Geometry
endif
LIBS := $(LINK) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/so_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
