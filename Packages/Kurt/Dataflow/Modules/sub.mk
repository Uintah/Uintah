#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Kurt/Modules

SUBDIRS := $(SRCDIR)/VolVis

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:40  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:33  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
