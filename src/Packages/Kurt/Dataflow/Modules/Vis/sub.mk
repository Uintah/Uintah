#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Kurt/Modules/Vis

SRCS     += $(SRCDIR)/GLTextureBuilder.cc  $(SRCDIR)/PadField.cc \
		$(SRCDIR)/TextureVolVis.cc \
		$(SRCDIR)/TexCuttingPlanes.cc \
		$(SRCDIR)/VolVis.cc \
		$(SRCDIR)/ArchiveReader.cc \
		$(SRCDIR)/KurtScalarFieldReader.cc \
		$(SRCDIR)/VisControl.cc \
		$(SRCDIR)/RescaleColorMapForParticles.cc \
		$(SRCDIR)/ParticleVis.cc



PSELIBS :=  PSECore/Dataflow PSECore/Datatypes \
        SCICore/Thread SCICore/Persistent SCICore/Exceptions \
        SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
        SCICore/Geom Uintah/Grid Uintah/Interface Uintah/Exceptions \
	SCICore/Geometry PSECore/Widgets PSECore/XMLUtil \
	Kurt/Datatypes Kurt/DataArchive Kurt/Geom 

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
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
