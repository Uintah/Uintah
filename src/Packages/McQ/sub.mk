#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/so_prologue.mk

SRCDIR := McQ
SUBDIRS := $(SRCDIR)/Euler $(SRCDIR)/GUI

include $(OBJTOP_ABS)/scripts/recurse.mk

ifeq ($(LARGESOS),yes)
PSELIBS := PSECore SCICore
else
PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Containers \
	SCICore/Geom SCICore/Datatypes SCICore/Thread \
	SCICore/Persistent SCICore/Exceptions SCICore/TclInterface
endif
LIBS :=

include $(OBJTOP_ABS)/scripts/so_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:37  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
