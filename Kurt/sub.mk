#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Kurt

SUBDIRS := $(SRCDIR)/GUI $(SRCDIR)/Geom $(SRCDIR)/Modules \
		$(SRCDIR)/Datatypes $(SRCDIR)/DataArchive


include $(SRCTOP)/scripts/recurse.mk

ifeq ($(LARGESOS),yes)
PSELIBS := PSECore SCICore
else
PSELIBS := PSECore/Datatypes PSECore/Widgets \
	PSECore/Dataflow SCICore/Containers \
	SCICore/TclInterface SCICore/Exceptions SCICore/Persistent \
	SCICore/Thread SCICore/Geom SCICore/Datatypes SCICore/Geometry
endif
LIBS := $(LINK) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/05/20 02:31:44  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.4  2000/05/16 20:53:59  kuzimmer
# added new directory
#
# Revision 1.3  2000/03/21 17:33:23  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:35  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
