#
# Makefile fragment for this subdirectory
# $Id$
#

#
# The SRCDIR_STACK maintains the current state of the recursion.
# This is used to reset SRCDIR after the include below is processed.
#
SRCDIR_STACK := $(SRCDIR) $(SRCDIR_STACK)
ALLSUBDIRS := $(ALLSUBDIRS) $(SUBDIRS)

TMPSUBDIRS := $(SUBDIRS)
SUBDIRS := SET SUBDIRS BEFORE CALLING RECURSE
include $(patsubst %,%/sub.mk,$(TMPSUBDIRS))

SRCDIR := $(firstword $(SRCDIR_STACK))
SRCDIR_STACK := $(wordlist 2,$(words $(SRCDIR_STACK)),$(SRCDIR_STACK))

#
# $Log$
# Revision 1.3  2000/03/17 09:30:57  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
