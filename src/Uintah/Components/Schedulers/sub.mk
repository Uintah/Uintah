#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/Schedulers

SRCS     += $(SRCDIR)/BrainDamagedScheduler.cc \
	$(SRCDIR)/OnDemandDataWarehouse.cc

PSELIBS := Uintah/Grid Uintah/Interface SCICore/Thread Uintah/Parallel \
	Uintah/Exceptions
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:29:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#