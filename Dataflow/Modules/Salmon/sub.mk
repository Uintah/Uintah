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
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Comm \
	SCICore/Persistent SCICore/Exceptions SCICore/Geometry \
	SCICore/Geom SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/TkExtensions SCICore/Util \
	SCICore/TkExtensions SCICore/Datatypes

LIBS := $(TK_LIBRARY) $(GL_LIBS) $(IMAGE_LIBS) -lm


include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.8  2000/07/18 23:11:11  samsonov
# EditPath module added
#
# Revision 1.7  2000/06/09 17:50:19  kuzimmer
# Hopefully everything is fixed so that you can use -lifl on SGI's and you can use -lcl on SGI's in32bit mode.
#
# Revision 1.6  2000/06/08 22:46:29  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.5  2000/06/08 20:32:01  kuzimmer
#  modified sub.mk so that linux will compile: you need to load new configure.in
#
# Revision 1.4  2000/06/07 20:59:27  kuzimmer
# Modifications to make the image save menu item work on SGIs
#
# Revision 1.3  2000/06/07 00:11:40  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:03  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:18  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
