#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms

SUBDIRS := $(SRCDIR)/Malloc $(SRCDIR)/Thread
ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/Component
endif

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:39:14  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:31:01  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
