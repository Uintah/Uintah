#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/ThirdParty

SUBDIRS := $(SRCDIR)/NumRec $(SRCDIR)/Nrrd $(SRCDIR)/OldLinAlg

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:27  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
