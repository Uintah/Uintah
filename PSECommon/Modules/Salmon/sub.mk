#
# Makefile fragment for this subdirectory
# $Id$
#

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
