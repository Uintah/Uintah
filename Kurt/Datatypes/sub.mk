#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Kurt/Datatypes

SRCS     += $(SRCDIR)/Brick.cc \
	 $(SRCDIR)/GLTexture3D.cc \
	 $(SRCDIR)/GLTexture3DPort.cc \
	 $(SRCDIR)/GLTextureIterator.cc \
	 $(SRCDIR)/GLTexRenState.cc \
	 $(SRCDIR)/GLAttenuate.cc \
	 $(SRCDIR)/GLOverOp.cc \
	 $(SRCDIR)/GLMIP.cc \
	 $(SRCDIR)/GLVolRenState.cc \
	 $(SRCDIR)/FullRes.cc \
	 $(SRCDIR)/FullResIterator.cc \
	 $(SRCDIR)/LOS.cc \
	 $(SRCDIR)/LOSIterator.cc \
	 $(SRCDIR)/ROI.cc \
	 $(SRCDIR)/ROIIterator.cc \
	 $(SRCDIR)/GLVolumeRenderer.cc \
	 $(SRCDIR)/Octree.cc \
	 $(SRCDIR)/Polygon.cc \
	 $(SRCDIR)/SliceTable.cc \
	 $(SRCDIR)/VolumeUtils.cc 



#
# $Log$
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
