#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Exceptions

SRCS     += $(SRCDIR)/DataWarehouseException.cc \
	$(SRCDIR)/ParticleException.cc $(SRCDIR)/ProblemSetupException.cc \
	$(SRCDIR)/SchedulerException.cc $(SRCDIR)/TypeMismatchException.cc

PSELIBS :=
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:29:54  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#