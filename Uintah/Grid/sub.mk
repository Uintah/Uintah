#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Grid

SRCS     += $(SRCDIR)/Array3Index.cc $(SRCDIR)/DataItem.cc \
	$(SRCDIR)/Grid.cc $(SRCDIR)/Level.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ProblemSpec.cc \
	$(SRCDIR)/RefCounted.cc $(SRCDIR)/Region.cc $(SRCDIR)/SubRegion.cc \
	$(SRCDIR)/Task.cc

PSELIBS := Uintah/Math Uintah/Exceptions SCICore/Thread SCICore/Exceptions
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:30:00  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#