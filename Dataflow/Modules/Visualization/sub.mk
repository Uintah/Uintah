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
	$(SRCDIR)/GLTextureBuilder.cc\
	$(SRCDIR)/GenStandardColorMaps.cc\
	$(SRCDIR)/GenTransferFunc.cc\
	$(SRCDIR)/Hedgehog.cc\
	$(SRCDIR)/HedgehogLitLines.cc\
	$(SRCDIR)/ImageViewer.cc\
	$(SRCDIR)/IsoSurface.cc\
	$(SRCDIR)/IsoSurfaceDW.cc\
	$(SRCDIR)/RescaleColorMap.cc\
	$(SRCDIR)/ScalarFieldProbe.cc\
	$(SRCDIR)/ShowColorMap.cc\
	$(SRCDIR)/ShowDipoles.cc\
	$(SRCDIR)/ShowField.cc\
	$(SRCDIR)/ShowFieldSlice.cc\
	$(SRCDIR)/ShowHist.cc\
	$(SRCDIR)/ShowImage.cc\
	$(SRCDIR)/ShowMatrix.cc\
	$(SRCDIR)/ShowMesh.cc\
	$(SRCDIR)/ShowSurface.cc\
	$(SRCDIR)/ShowWidgets.cc\
	$(SRCDIR)/Streamline.cc\
	$(SRCDIR)/TexCuttingPlanes.cc\
	$(SRCDIR)/TextureVolVis.cc\
#	$(SRCDIR)/FieldCage.cc\
#	$(SRCDIR)/Ted.cc\
[INSERT NEW CODE FILE HERE]


PSELIBS := Core/Algorithms/Visualization \
	Dataflow/Network Core/Datatypes Dataflow/Widgets \
	Core/Containers Core/Exceptions Core/Thread \
	Core/TclInterface Core/Geom Core/Persistent \
	Dataflow/Ports Core/Geometry Core/Util \
	Core/TkExtensions Dataflow/Modules/Render
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
