#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Visualization

SRCS     += $(SRCDIR)/AddWells.cc $(SRCDIR)/AddWells2.cc \
	$(SRCDIR)/BitVisualize.cc $(SRCDIR)/BoundGrid.cc \
	$(SRCDIR)/BoxClipSField.cc $(SRCDIR)/ColorMapKey.cc \
	$(SRCDIR)/CuttingPlane.cc $(SRCDIR)/CuttingPlaneTex.cc \
	$(SRCDIR)/FieldCage.cc $(SRCDIR)/GenAxes.cc \
	$(SRCDIR)/GenColorMap.cc $(SRCDIR)/GenFieldEdges.cc \
	$(SRCDIR)/GenStandardColorMaps.cc $(SRCDIR)/GenTransferFunc.cc \
	$(SRCDIR)/Hedgehog.cc $(SRCDIR)/ImageViewer.cc $(SRCDIR)/IsoMask.cc \
	$(SRCDIR)/IsoSurface.cc $(SRCDIR)/IsoSurfaceDW.cc \
	$(SRCDIR)/IsoSurfaceMRSG.cc $(SRCDIR)/IsoSurfaceSP.cc \
	$(SRCDIR)/RescaleColorMap.cc $(SRCDIR)/SimpVolVis.cc \
	$(SRCDIR)/Streamline.cc $(SRCDIR)/VectorSeg.cc \
	$(SRCDIR)/VolRendTexSlices.cc $(SRCDIR)/VolVis.cc \
	$(SRCDIR)/WidgetTest.cc $(SRCDIR)/FastRender.c \
	$(SRCDIR)/HedgehogLitLines.cc $(SRCDIR)/Span.cc \
	$(SRCDIR)/Noise.cc $(SRCDIR)/NoiseMCube.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Containers SCICore/Exceptions SCICore/Thread \
	SCICore/TclInterface SCICore/Geom SCICore/Persistent \
	SCICore/Datatypes SCICore/Geometry SCICore/Util \
	SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:06  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:37  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
