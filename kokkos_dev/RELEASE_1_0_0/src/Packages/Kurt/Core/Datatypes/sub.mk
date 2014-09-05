# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Kurt/Core/Datatypes

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

PSELIBS := Core/Exceptions Core/Geometry \
	Core/Persistent Core/Datatypes \
	Core/Containers  Core/Geom Core/Thread \
	Dataflow/Network PSECore/XMLUtil \
	Uintah/Grid Packages/Uintah/Core/Datatypes Uintah/Exceptions \
	Packages/Uintah/Dataflow/Modules/Visualization

LIBS :=  $(LINK) $(XML_LIBRARY) $(GL_LIBS) -lmpi -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

