#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Yarden/Modules/Visualization

SRCS     += $(SRCDIR)/Sage.cc $(SRCDIR)/Screen.cc 

PSELIBS := Yarden/Datatypes/General PSECore/Datatypes PSECore/Dataflow \
	SCICore/Persistent SCICore/Containers SCICore/Util \
	SCICore/Exceptions SCICore/Thread SCICore/TclInterface \
	SCICore/Geom SCICore/Datatypes SCICore/Geometry \
	SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/20 23:38:40  yarden
# Linux port
#
# Revision 1.2  2000/03/20 19:38:58  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:37  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
