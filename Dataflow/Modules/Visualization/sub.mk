#
# Makefile fragment for this subdirectory
# $Id$
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Visualization

SRCS     += \
	$(SRCDIR)/AddWells.cc\
	$(SRCDIR)/AddWells2.cc\
	$(SRCDIR)/BitVisualize.cc\
	$(SRCDIR)/BoundGrid.cc\
	$(SRCDIR)/BoxClipSField.cc\
	$(SRCDIR)/ColorMapKey.cc\
	$(SRCDIR)/CuttingPlane.cc\
	$(SRCDIR)/CuttingPlaneTex.cc\
	$(SRCDIR)/FieldCage.cc\
	$(SRCDIR)/GenAxes.cc\
	$(SRCDIR)/GenColorMap.cc\
	$(SRCDIR)/GenFieldEdges.cc\
	$(SRCDIR)/GenStandardColorMaps.cc\
	$(SRCDIR)/GenTransferFunc.cc\
	$(SRCDIR)/Hedgehog.cc\
	$(SRCDIR)/ImageViewer.cc\
	$(SRCDIR)/IsoMask.cc\
	$(SRCDIR)/IsoSurface.cc\
	$(SRCDIR)/IsoSurfaceDW.cc\
	$(SRCDIR)/IsoSurfaceMRSG.cc\
	$(SRCDIR)/IsoSurfaceMSRG.cc\
	$(SRCDIR)/IsoSurfaceSAGE.cc\
	$(SRCDIR)/IsoSurfaceNOISE.cc\
	$(SRCDIR)/Span.cc\
	$(SRCDIR)/SearchNOISE.cc\
	$(SRCDIR)/RescaleColorMap.cc\
	$(SRCDIR)/SimpVolVis.cc\
	$(SRCDIR)/Streamline.cc\
	$(SRCDIR)/VectorSeg.cc\
	$(SRCDIR)/VolRendTexSlices.cc\
	$(SRCDIR)/VolVis.cc\
	$(SRCDIR)/WidgetTest.cc\
	$(SRCDIR)/FastRender.c\
	$(SRCDIR)/HedgehogLitLines.cc\
        $(SRCDIR)/GLTextureBuilder.cc\
        $(SRCDIR)/TextureVolVis.cc\
        $(SRCDIR)/TexCuttingPlanes.cc\
#[INSERT NEW CODE FILE HERE]

#	$(SRCDIR)/Span.cc\
#	$(SRCDIR)/Noise.cc\
#	$(SRCDIR)/NoiseMCube.cc\

PSELIBS := PSECommon/Algorithms/Visualization \
	PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Containers SCICore/Exceptions SCICore/Thread \
	SCICore/TclInterface SCICore/Geom SCICore/Persistent \
	SCICore/Datatypes SCICore/Geometry SCICore/Util \
	SCICore/TkExtensions PSECommon/Modules/Salmon
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.11  2000/12/09 17:29:50  moulding
# Port Kurt's volume rendering stuff to linux and move it to PSECommon.
#
# Revision 1.10  2000/11/30 17:58:29  moulding
# moved IsoSurfaceSP -> IsoSurfaceMSRG
#
# Revision 1.9  2000/10/24 05:57:39  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.8  2000/07/26 20:14:46  yarden
# Isosurface extraction module based on the NOISE
# algorithm. This module accepts SpanUniverse through
# a port rather then create one based on the given
# scalar field.
#
# Revision 1.7  2000/07/24 20:56:29  yarden
# A new module to extract isosurfaces base on Noise algorithm
# works on UG as well as any RG type
#
# Revision 1.6  2000/07/22 18:01:39  yarden
# add IsoSurfaceSAGE
#
# Revision 1.5  2000/06/08 22:46:31  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.4  2000/06/07 00:11:41  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.3  2000/03/20 21:49:04  yarden
# Linux port: link against PSECommon/Modules/Salmon too.
#
# Revision 1.2  2000/03/20 19:37:06  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:37  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
