#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Uintah/Modules

SUBDIRS := \
	$(SRCDIR)/Readers\
	$(SRCDIR)/Visualization\
#[INSERT NEW CATEGORY DIR HERE]


include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.6  2000/12/07 18:47:00  kuzimmer
# not building a writers directory since we currently have no writers.
#
# Revision 1.5  2000/12/06 21:50:57  kuzimmer
# No longer compiling MPMVis
#
# Revision 1.4  2000/10/24 05:57:55  moulding
#
#
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/06/20 17:57:16  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.2  2000/03/20 19:38:40  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
