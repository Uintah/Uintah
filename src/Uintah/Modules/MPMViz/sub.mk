#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/MPMViz

SRCS     += \
	$(SRCDIR)/ParticleGridVisControl.cc\
	$(SRCDIR)/PartToGeom.cc \
	$(SRCDIR)/RescaleParticleColorMap.cc\
	$(SRCDIR)/cfdGridLines.cc \
	$(SRCDIR)/GridLines.cc\
	$(SRCDIR)/TecplotFileSelector.cc \
	$(SRCDIR)/VizControl.cc\
#[INSERT NEW CODE FILE HERE]

ifeq ($(BUILD_PARALLEL),yes)
SRCS += $(SRCDIR)/ParticleDB.cc $(SRCDIR)/ParticleDatabase.cc \
	$(SRCDIR)/ParticleViz.cc $(SRCDIR)/RunSimulation.cc
GENHDRS := Uintah/Datatypes/Particles/Particles_sidl.h
endif

PSELIBS := Uintah/Datatypes PSECore/Dataflow PSECore/Datatypes \
	SCICore/Thread SCICore/Persistent SCICore/Exceptions \
	SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
	SCICore/Geom
LIBS := -lm
ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Component/PIDL Component/CIA
LIBS := $(LIBS)
endif

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4.2.1  2000/10/26 10:06:20  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.5  2000/10/24 05:57:56  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.4  2000/06/20 17:57:17  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.3  2000/03/23 11:18:16  sparker
# Makefile tweaks for sidl files
# Added GENHDRS to program.mk
#
# Revision 1.2  2000/03/20 19:38:42  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:13  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
