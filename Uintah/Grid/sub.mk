#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Grid

SRCS     += $(SRCDIR)/Array3Index.cc $(SRCDIR)/DataItem.cc \
	$(SRCDIR)/Grid.cc $(SRCDIR)/Level.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc \
	$(SRCDIR)/RefCounted.cc $(SRCDIR)/Region.cc $(SRCDIR)/SubRegion.cc \
	$(SRCDIR)/Task.cc $(SRCDIR)/Material.cc

PSELIBS := Uintah/Math Uintah/Exceptions SCICore/Thread SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/03/30 18:28:52  guilkey
# Moved Material class into Grid directory.  Put indices to velocity
# field and data warehouse into the base class.
#
# Revision 1.3  2000/03/23 20:42:22  sparker
# Added copy ctor to exception classes (for Linux/g++)
# Helped clean up move of ProblemSpec from Interface to Grid
#
# Revision 1.2  2000/03/20 19:38:35  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:00  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
