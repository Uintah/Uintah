#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Exceptions

SRCS     += $(SRCDIR)/DataWarehouseException.cc \
	$(SRCDIR)/ParticleException.cc $(SRCDIR)/ProblemSetupException.cc \
	$(SRCDIR)/SchedulerException.cc $(SRCDIR)/TypeMismatchException.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/23 10:30:30  sparker
# Update to use new Exception base class
#
# Revision 1.2  2000/03/20 19:38:32  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:54  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
