#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/MPMViz

SRCS     += $(SRCDIR)/ParticleGridVisControl.cc $(SRCDIR)/PartToGeom.cc \
	$(SRCDIR)/RescaleParticleColorMap.cc $(SRCDIR)/cfdGridLines.cc \
	$(SRCDIR)/GridLines.cc $(SRCDIR)/TecplotFileSelector.cc \
	$(SRCDIR)/VizControl.cc

ifeq ($(BUILD_PARALLEL),yes)
SRCS += $(SRCDIR)/ParticleDB.cc $(SRCDIR)/ParticleDatabase.cc \
	$(SRCDIR)/ParticleViz.cc $(SRCDIR)/RunSimulation.cc
endif

PSELIBS := Uintah/Datatypes/Particles PSECore/Dataflow PSECore/Datatypes \
	SCICore/Thread SCICore/Persistent SCICore/Exceptions \
	SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
	SCICore/Geom
LIBS := -lm
ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Component/PIDL Component/CIA
LIBS := $(LIBS)
endif

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:30:13  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
