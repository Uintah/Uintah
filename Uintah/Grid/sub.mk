#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Grid

SRCS     += $(SRCDIR)/Array3Index.cc $(SRCDIR)/Box.cc \
	$(SRCDIR)/CellIterator.cc $(SRCDIR)/CCVariableBase.cc \
	$(SRCDIR)/Grid.cc \
	$(SRCDIR)/Level.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/NCVariableBase.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ParticleVariableBase.cc \
	$(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/RefCounted.cc $(SRCDIR)/Region.cc \
	$(SRCDIR)/SimulationState.cc \
	$(SRCDIR)/SimulationTime.cc $(SRCDIR)/SubRegion.cc \
	$(SRCDIR)/Task.cc $(SRCDIR)/TypeDescription.cc \
	$(SRCDIR)/TypeUtils.cc $(SRCDIR)/VarLabel.cc \
	$(SRCDIR)/templates.cc

PSELIBS := Uintah/Math Uintah/Exceptions SCICore/Thread SCICore/Exceptions \
	SCICore/Geometry
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.14  2000/05/20 08:09:31  sparker
# Improved TypeDescription
# Finished I/O
# Use new XML utility libraries
#
# Revision 1.13  2000/05/12 18:12:37  sparker
# Added CCVariableBase.cc to sub.mk
# Fixed copyPointer and other CCVariable methods - still not implemented
#
# Revision 1.12  2000/05/11 20:10:22  dav
# adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
#
# Revision 1.11  2000/05/07 06:02:14  sparker
# Added beginnings of multiple patch support and real dependencies
#  for the scheduler
#
# Revision 1.10  2000/04/28 03:58:20  sparker
# Fixed countParticles
# Implemented createParticles, which doesn't quite work yet because the
#   data warehouse isn't there yet.
# Reduced the number of particles in the bar problem so that it will run
#   quickly during development cycles
#
# Revision 1.9  2000/04/20 22:58:20  sparker
# Resolved undefined symbols
# Trying to make stuff work
#
# Revision 1.8  2000/04/20 22:37:17  jas
# Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
# to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
#
# Revision 1.7  2000/04/20 18:56:32  sparker
# Updates to MPM
#
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
