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
	$(SRCDIR)/RescaleColorMap.cc\
	$(SRCDIR)/ShowField.cc\
	$(SRCDIR)/Streamline.cc\
	$(SRCDIR)/TexCuttingPlanes.cc\
	$(SRCDIR)/TextureVolVis.cc\
	$(SRCDIR)/Isosurface.cc\
[INSERT NEW CODE FILE HERE]




PSELIBS := Dataflow/Network Core/Datatypes Dataflow/Widgets \
	Core/Containers Core/Exceptions Core/Thread \
	Core/GuiInterface Core/Geom Core/Persistent \
	Dataflow/Ports Core/Geometry Core/Util \
	Core/TkExtensions Dataflow/Modules/Render
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
