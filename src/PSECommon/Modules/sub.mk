#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := PSECommon/Modules

SUBDIRS := $(SRCDIR)/Example $(SRCDIR)/FEM $(SRCDIR)/Fields \
	$(SRCDIR)/Iterators $(SRCDIR)/Matrix $(SRCDIR)/Readers \
	$(SRCDIR)/Salmon $(SRCDIR)/Surface $(SRCDIR)/Visualization \
	$(SRCDIR)/Writers

include $(OBJTOP_ABS)/scripts/recurse.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:46  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
