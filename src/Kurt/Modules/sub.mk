#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Kurt/Modules

SUBDIRS := \
	$(SRCDIR)/Vis\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3.2.2  2000/10/26 13:24:50  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4  2000/10/24 05:57:21  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/05/20 02:31:29  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.2  2000/03/20 19:36:40  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:33  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
