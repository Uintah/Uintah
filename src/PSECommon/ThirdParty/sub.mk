#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := PSECommon/ThirdParty

SUBDIRS := $(SRCDIR)/mpeg \
	   $(SRCDIR)/mpeg_encode 

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3  2001/01/17 20:09:25  kuzimmer
# added Berkely mpeg_encode library
#
# Revision 1.2  2000/03/20 19:37:09  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
