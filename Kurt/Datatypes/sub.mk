#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Kurt/Datatypes

SRCS     += $(SRCDIR)/Brick.cc \
	 $(SRCDIR)/GLTexture3D.cc \
	 $(SRCDIR)/GLTexture3DPort.cc \
	 $(SRCDIR)/GLTextureIterator.cc \
	 $(SRCDIR)/GLTexRenState.cc \
	 $(SRCDIR)/GLAttenuate.cc \
	 $(SRCDIR)/GLOverOp.cc \
	 $(SRCDIR)/GLMIP.cc \
	 $(SRCDIR)/GLPlanes.cc \
	 $(SRCDIR)/GLVolRenState.cc \
	 $(SRCDIR)/LevelIterator.cc \
	 $(SRCDIR)/FullRes.cc \
	 $(SRCDIR)/FullResIterator.cc \
	 $(SRCDIR)/LOS.cc \
	 $(SRCDIR)/LOSIterator.cc \
	 $(SRCDIR)/ROI.cc \
	 $(SRCDIR)/ROIIterator.cc \
	 $(SRCDIR)/TexPlanes.cc \
	 $(SRCDIR)/GLVolumeRenderer.cc \
	 $(SRCDIR)/Octree.cc \
	 $(SRCDIR)/Polygon.cc \
	 $(SRCDIR)/SliceTable.cc \
	 $(SRCDIR)/VolumeUtils.cc \
	 $(SRCDIR)/VisParticleSet.cc \
	 $(SRCDIR)/VisParticleSetPort.cc \
	 $(SRCDIR)/GLAnimatedStreams.cc

PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers  SCICore/Geom SCICore/Thread \
	PSECore/Dataflow PSECore/XMLUtil \
	Uintah/Grid Uintah/Datatypes Uintah/Exceptions \
	Uintah/Modules/Visualization



LIBS :=  $(LINK) $(XML_LIBRARY) $(GL_LIBS) -lmpi -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.8  2001/01/05 17:36:27  kuzimmer
# Implemented multithreaded texture building and cleaned up some files.
#
# Revision 1.7  2000/12/06 04:43:02  kuzimmer
# Added PSECore/Dataflow and PSECore/XMLUtil to PSELIBS.  Added -lmpi to LIBS.  To remove unresolved symbol warnings on compile
#
# Revision 1.6  2000/09/27 16:23:18  kuzimmer
# Moved these files from the now defunct DataArchive Directory
#
# Revision 1.5  2000/09/17 15:59:41  kuzimmer
# updated texture planes for binary transparency
#
# Revision 1.4  2000/08/29 21:19:43  kuzimmer
# Added some 3D texture mapping functionality
#
# Revision 1.3  2000/05/29 22:24:49  kuzimmer
# A bunch of fixes, including making volumes work with udas, transforming volumes properly without copying data, and fixing coredumps when changing udas
#
# Revision 1.2  2000/05/20 02:23:28  kuzimmer
# modifications for a texture slicing module
#
# Revision 1.1  2000/05/16 20:52:39  kuzimmer
# files for new volume renderer
#
# Revision 1.3  2000/03/21 17:33:26  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:38  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:31  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
