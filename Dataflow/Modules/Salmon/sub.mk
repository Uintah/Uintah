#
# Makefile fragment for this subdirectory
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

SRCDIR   := PSECommon/Modules/Salmon

SRCS     += \
	$(SRCDIR)/Ball.cc\
	$(SRCDIR)/BallAux.cc\
	$(SRCDIR)/BallMath.cc\
	$(SRCDIR)/Tex.cc\
	$(SRCDIR)/close.c\
	$(SRCDIR)/name.c\
	$(SRCDIR)/open.c\
	$(SRCDIR)/rdwr.c\
	$(SRCDIR)/rle.c\
	$(SRCDIR)/MpegEncoder.cc\
	$(SRCDIR)/row.c\
	$(SRCDIR)/Roe.cc\
	$(SRCDIR)/OpenGL.cc\
	$(SRCDIR)/Renderer.cc\
	$(SRCDIR)/Salmon.cc\
	$(SRCDIR)/SalmonGeom.cc\
	$(SRCDIR)/BaWGL.cc\
	$(SRCDIR)/Parser.cc\
	$(SRCDIR)/SCIBaWGL.cc\
	$(SRCDIR)/SharedMemory.cc\
	$(SRCDIR)/glMath.cc\
	$(SRCDIR)/EditPath.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Widgets PSECore/Dataflow PSECore/Datatypes PSECore/Comm \
	SCICore/Persistent SCICore/Exceptions SCICore/Geometry \
	SCICore/Geom SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/TkExtensions SCICore/Util \
	SCICore/TkExtensions SCICore/Datatypes

LIBS := $(TK_LIBRARY) $(GL_LIBS) $(IMAGE_LIBS) -lm


include $(SRCTOP)/scripts/smallso_epilogue.mk

