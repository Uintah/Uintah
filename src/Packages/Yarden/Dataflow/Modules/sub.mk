#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Yarden/Modules

SUBDIRS := \
	$(SRCDIR)/Readers \
	$(SRCDIR)/Visualization \
	$(SRCDIR)/Writers


include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3  2000/10/23 23:43:44  yarden
# initial commit
#
# Revision 1.2  2000/03/20 19:38:56  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:33  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
