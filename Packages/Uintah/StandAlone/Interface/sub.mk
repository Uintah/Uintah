#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Interface

SRCS     += $(SRCDIR)/CFDInterface.cc $(SRCDIR)/DataWarehouse.cc \
	$(SRCDIR)/MPMInterface.cc $(SRCDIR)/Output.cc \
	$(SRCDIR)/Scheduler.cc

PSELIBS := Uintah/Parallel Uintah/Grid
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:30:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#