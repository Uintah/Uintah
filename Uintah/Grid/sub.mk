#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Grid

SRCS     += $(SRCDIR)/Array3Index.cc $(SRCDIR)/Box.cc \
	$(SRCDIR)/DataItem.cc $(SRCDIR)/Grid.cc \
	$(SRCDIR)/Level.cc $(SRCDIR)/Material.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/RefCounted.cc $(SRCDIR)/Region.cc \
	$(SRCDIR)/SimulationTime.cc $(SRCDIR)/SubRegion.cc \
	$(SRCDIR)/Task.cc $(SRCDIR)/VarLabel.cc

PSELIBS := Uintah/Math Uintah/Exceptions SCICore/Thread SCICore/Exceptions \
	SCICore/Geometry
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.6  2000/04/19 05:26:15  sparker
# Implemented new problemSetup/initialization phases
# Simplified DataWarehouse interface (not finished yet)
# Made MPM get through problemSetup, but still not finished
#
# Revision 1.5  2000/04/13 06:51:03  sparker
# More implementation to get this to work
#
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
