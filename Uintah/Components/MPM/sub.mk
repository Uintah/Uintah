#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc $(SRCDIR)/ThreadedMPM.cc \
	$(SRCDIR)/BoundCond.cc

SUBDIRS := $(SRCDIR)/ConstitutiveModel $(SRCDIR)/Contact \
	$(SRCDIR)/GeometrySpecification $(SRCDIR)/Util

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:29:32  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#