#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Phil/Modules

SUBDIRS := \
	$(SRCDIR)/Tbon\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2.2.2  2000/10/26 13:37:11  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.3  2000/10/24 05:57:43  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.2  2000/03/20 19:37:28  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
