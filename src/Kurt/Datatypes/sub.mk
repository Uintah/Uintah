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
	 $(SRCDIR)/VolumeUtils.cc 

PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers  SCICore/Geom SCICore/Thread \
	Uintah/Grid Uintah/Datatypes Uintah/Exceptions \
	Uintah/Modules/Visualization



LIBS :=  $(LINK) $(XML_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
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
