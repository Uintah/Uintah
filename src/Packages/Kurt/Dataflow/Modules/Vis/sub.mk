#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Kurt/Modules/Vis

SRCS     += \
	$(SRCDIR)/GLTextureBuilder.cc\
	$(SRCDIR)/PadField.cc \
	$(SRCDIR)/TextureVolVis.cc \
	$(SRCDIR)/TexCuttingPlanes.cc \
	$(SRCDIR)/ParticleColorMapKey.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/AnimatedStreams.cc \
#[INSERT NEW CODE FILE HERE]

#		$(SRCDIR)/VolVis.cc \
#		$(SRCDIR)/KurtScalarFieldReader.cc \
#		$(SRCDIR)/VisControl.cc \
#		$(SRCDIR)/ParticleVis.cc \


PSELIBS :=  PSECore/Dataflow PSECore/Datatypes \
        SCICore/Thread SCICore/Persistent SCICore/Exceptions \
        SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
        SCICore/Geom SCICore/Geometry PSECore/Widgets PSECore/XMLUtil \
	Kurt/Datatypes SCICore/Util \
	Uintah/Datatypes Uintah/Grid Uintah/Interface Uintah/Exceptions 


LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
# Revision 1.4.2.2  2000/10/26 10:03:06  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4.2.1  2000/09/28 03:18:07  mcole
# merge trunk into FIELD_REDESIGN branch
#
# Revision 1.15  2000/10/24 05:57:22  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.14  2000/09/27 16:24:08  kuzimmer
# changes made to reflect the moved VisParticleSet files
#
# Revision 1.13  2000/09/26 19:02:45  kuzimmer
# to remove dependency on libKurt_Geom.so
#
# Revision 1.12  2000/09/26 18:24:39  kuzimmer
# to remove dependency on libKurt_DataArchive.so
#
# Revision 1.11  2000/09/21 22:22:49  kuzimmer
# some lines that weren't supposed to be commented out were.  fixed.
#
# Revision 1.10  2000/09/20 22:47:13  kuzimmer
# changes so that the Kurt subtree can be compiled without Uintah by commenting out a few lines.
#
# Revision 1.9  2000/09/17 16:05:07  kuzimmer
# C++ code for animated streams
#
# Revision 1.8  2000/06/21 04:14:16  kuzimmer
# removed unneeded dependencies on Kurt
#
# Revision 1.7  2000/06/15 19:49:39  sparker
# Link against SCICore/Util
#
# Revision 1.6  2000/06/13 20:28:16  kuzimmer
# Added a colormap key sticky for particle sets
#
# Revision 1.5  2000/06/05 21:10:31  bigler
# Added new module to visualize UINTAH grid
#
# Revision 1.4  2000/05/25 18:24:24  kuzimmer
# removing old volvis directory
#
# Revision 1.3  2000/05/24 00:25:26  kuzimmer
# Jim wants these updates so he can run Kurts stuff out of Steves 64 bit PSE. WARNING, I (dav) do not know quite what I do.
#
# Revision 1.2  2000/05/21 01:19:20  kuzimmer
# Added Archive Reader
#
# Revision 1.1  2000/05/20 02:30:09  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.4  2000/05/16 20:54:03  kuzimmer
# added new directory
#
# Revision 1.3  2000/03/21 17:33:28  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:41  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:35  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
