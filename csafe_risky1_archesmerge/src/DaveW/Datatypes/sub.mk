#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/Datatypes

SUBDIRS := $(SRCDIR)/CS684 $(SRCDIR)/General

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:35:53  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:18  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
