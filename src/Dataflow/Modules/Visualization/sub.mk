# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/AddWells.cc\
	$(SRCDIR)/AddWells2.cc\
	$(SRCDIR)/BitVisualize.cc\
	$(SRCDIR)/BoundGrid.cc\
	$(SRCDIR)/BoxClipSField.cc\
	$(SRCDIR)/ColorMapKey.cc\
	$(SRCDIR)/CuttingPlane.cc\
	$(SRCDIR)/CuttingPlaneTex.cc\
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
	$(SRCDIR)/Isosurface.cc\
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

#	$(SRCDIR)/IsoSurfaceSAGE.cc\
#	$(SRCDIR)/IsoSurfaceNOISE.cc\
#	$(SRCDIR)/FieldCage.cc\

#	$(SRCDIR)/Span.cc\
#	$(SRCDIR)/Noise.cc\
#	$(SRCDIR)/NoiseMCube.cc\


PSELIBS := Core/Algorithms/Visualization \
	Dataflow/Network Core/Datatypes Dataflow/Widgets \
	Core/Containers Core/Exceptions Core/Thread \
	Core/TclInterface Core/Geom Core/Persistent \
	Dataflow/Ports Core/Geometry Core/Util \
	Core/TkExtensions Dataflow/Modules/Salmon
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
