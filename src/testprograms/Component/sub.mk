#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms/Component

SUBDIRS := $(SRCDIR)/argtest $(SRCDIR)/memstress $(SRCDIR)/mitest \
	$(SRCDIR)/objects $(SRCDIR)/pingpong

include $(OBJTOP_ABS)/scripts/recurse.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:31:02  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
