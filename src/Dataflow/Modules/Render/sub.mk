# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Render

SRCS     += \
	$(SRCDIR)/BaWGL.cc\
	$(SRCDIR)/Ball.cc\
	$(SRCDIR)/BallAux.cc\
	$(SRCDIR)/BallMath.cc\
	$(SRCDIR)/EditPath.cc\
	$(SRCDIR)/MpegEncoder.cc\
	$(SRCDIR)/OpenGL.cc\
	$(SRCDIR)/Parser.cc\
	$(SRCDIR)/Renderer.cc\
	$(SRCDIR)/SCIBaWGL.cc\
	$(SRCDIR)/SharedMemory.cc\
	$(SRCDIR)/Tex.cc\
	$(SRCDIR)/ViewGeom.cc\
	$(SRCDIR)/ViewWindow.cc\
	$(SRCDIR)/Viewer.cc\
	$(SRCDIR)/glMath.cc\
	$(SRCDIR)/close.c\
	$(SRCDIR)/name.c\
	$(SRCDIR)/open.c\
	$(SRCDIR)/rdwr.c\
	$(SRCDIR)/rle.c\
	$(SRCDIR)/row.c\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Widgets Dataflow/Network Dataflow/Ports Core/Datatypes \
	Dataflow/Comm Core/Persistent Core/Exceptions Core/Geometry \
	Core/Geom Core/Thread Core/Containers \
	Core/TclInterface Core/TkExtensions Core/Util \
	Core/TkExtensions Core/Datatypes

LIBS := $(TK_LIBRARY) $(GL_LIBS) $(IMAGE_LIBS) -lm


include $(SRCTOP)/scripts/smallso_epilogue.mk

