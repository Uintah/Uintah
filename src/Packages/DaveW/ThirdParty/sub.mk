#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/ThirdParty

SUBDIRS := $(SRCDIR)/Nrrd $(SRCDIR)/NumRec $(SRCDIR)/OldLinAlg

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3  2000/10/29 04:05:01  dmw
# updates to make downhill simplex more interactive
#
# Revision 1.2  2000/03/20 19:36:27  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
