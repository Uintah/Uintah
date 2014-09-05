#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Remote/Tools

SUBDIRS := $(SRCDIR)/Util $(SRCDIR)/Math $(SRCDIR)/Model $(SRCDIR)/Image

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := -lm -L/usr/local/gnu/lib -lfl -ltiff -ljpeg

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/07/10 20:44:19  dahart
# initial commit
#
# Revision 1.1  2000/06/06 15:29:22  dahart
# - Added a new package / directory tree for the remote visualization
# framework.
# - Added a new Salmon-derived module with a small server-daemon,
# network support, and a couple of very basic remote vis. services.
#
# Revision 1.2  2000/03/20 19:36:49  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
