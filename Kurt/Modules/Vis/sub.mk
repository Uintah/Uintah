#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Kurt/Modules/Vis

SRCS     += $(SRCDIR)/GLTextureBuilder.cc  $(SRCDIR)/PadField.cc \
		$(SRCDIR)/TexCuttingPlanes.cc \
		$(SRCDIR)/TextureVolVis.cc \
		$(SRCDIR)/VolVis.cc \
		$(SRCDIR)/VisControl.cc \
		$(SRCDIR)/RescaleColorMapForParticles.cc \
		$(SRCDIR)/ParticleVis.cc

PSELIBS :=  PSECore/Dataflow PSECore/Datatypes \
        SCICore/Thread SCICore/Persistent SCICore/Exceptions \
        SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
        SCICore/Geom Uintah/Grid Uintah/Interface Uintah/Exceptions \
	SCICore/Geometry PSECore/Widgets \
	Kurt/Datatypes Kurt/DataArchive Kurt/Geom 

LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
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
