#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Datatypes/Particles

SRCS     += $(SRCDIR)/MPMaterial.cc $(SRCDIR)/MPRead.cc \
	$(SRCDIR)/MPVizParticleSet.cc $(SRCDIR)/ParticleGridReader.cc \
	$(SRCDIR)/ParticleGridReaderPort.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSetPort.cc $(SRCDIR)/cfdlibParticleSet.cc \
	$(SRCDIR)/VizGrid.cc $(SRCDIR)/MPVizGrid.cc $(SRCDIR)/MPWrite.cc \
	$(SRCDIR)/MPParticleGridReader.cc $(SRCDIR)/TecplotReader.cc \
	$(SRCDIR)/MFMPParticleGridReader.cc

ifeq ($(BUILD_PARALLEL),yes)
SRCS += $(SRCDIR)/Particles_sidl.cc $(SRCDIR)/PIDLObject.cc \
	$(SRCDIR)/PIDLObjectPort.cc
GENHDRS := $(SRCDIR)/Particles_sidl.h
endif

#
# $Log$
# Revision 1.5  2000/06/20 20:13:38  kuzimmer
# updated so that Archiver reader will compile in Uintah
#
# Revision 1.4  2000/03/23 11:18:15  sparker
# Makefile tweaks for sidl files
# Added GENHDRS to program.mk
#
# Revision 1.3  2000/03/21 06:13:33  sparker
# Added pattern rule for .sidl files
# Compile component testprograms
#
# Revision 1.2  2000/03/20 19:38:31  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:51  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
