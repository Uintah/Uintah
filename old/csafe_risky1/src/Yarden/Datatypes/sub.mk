#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Yarden/Datatypes

SUBDIRS := $(SRCDIR)/General 

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:38:52  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
