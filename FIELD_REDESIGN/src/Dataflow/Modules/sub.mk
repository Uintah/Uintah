#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := PSECommon/Modules

SUBDIRS := $(SRCDIR)/Example $(SRCDIR)/FEM $(SRCDIR)/Fields \
	$(SRCDIR)/Iterators $(SRCDIR)/Matrix $(SRCDIR)/Readers \
	$(SRCDIR)/Salmon $(SRCDIR)/Surface $(SRCDIR)/Visualization \
	$(SRCDIR)/Writers $(SRCDIR)/Domain

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2.2.1  2000/06/07 17:23:37  kuehne
# Revised sub.mk to include Domain modules
#
# Revision 1.2  2000/03/20 19:36:52  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:46  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
