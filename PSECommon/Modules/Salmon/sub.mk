#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Salmon

SRCS     += $(SRCDIR)/Ball.cc $(SRCDIR)/BallAux.cc $(SRCDIR)/BallMath.cc \
	$(SRCDIR)/Tex.cc $(SRCDIR)/close.c $(SRCDIR)/name.c \
	$(SRCDIR)/open.c $(SRCDIR)/rdwr.c $(SRCDIR)/rle.c \
	$(SRCDIR)/MpegEncoder.cc $(SRCDIR)/row.c $(SRCDIR)/Roe.cc \
	$(SRCDIR)/OpenGL.cc $(SRCDIR)/Renderer.cc $(SRCDIR)/Salmon.cc \
	$(SRCDIR)/SalmonGeom.cc $(SRCDIR)/BaWGL.cc $(SRCDIR)/Parser.cc \
	$(SRCDIR)/SCIBaWGL.cc $(SRCDIR)/SharedMemory.cc $(SRCDIR)/glMath.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Comm \
	SCICore/Persistent SCICore/Exceptions SCICore/Geometry \
	SCICore/Geom SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/TkExtensions SCICore/Util \
	SCICore/TkExtensions SCICore/Datatypes
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:27:18  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
